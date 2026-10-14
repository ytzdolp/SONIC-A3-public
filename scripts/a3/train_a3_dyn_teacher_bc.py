#!/usr/bin/env python3
"""Bootstrap the native A3 dynamic decoder from successful teacher rollouts.

The Stage-1 A3 encoder and FSQ tokenizer are frozen.  Only ``a3_dyn`` is
optimized to predict the physical teacher's 29-DoF position action from the
64-D motion token and the 10-frame (930-D) SONIC proprioception history.

For a meaningful validation score, pass whole held-out clips through
``--validation-feature-dataset``. Random frame splitting remains available
only as a fallback for legacy experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from vector_quantize_pytorch import FSQ


ENC_PREFIX = "actor_module.encoders.a3.module."
DYN_PREFIX = "actor_module.decoders.a3_dyn.module."
HISTORY = 10
FUTURE = 10
DT_FUTURE = 0.1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(din, dout))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def load_prefixed(module: nn.Module, state: dict, prefix: str) -> None:
    selected = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    missing, unexpected = module.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch prefix={prefix}: missing={missing} unexpected={unexpected}")


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def relative_rot6d(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    inv = current.copy()
    inv[1:] *= -1.0
    rel = quat_mul(np.broadcast_to(inv, future.shape), future)
    rel /= np.linalg.norm(rel, axis=-1, keepdims=True).clip(1.0e-8)
    w, x, y, z = np.moveaxis(rel, -1, 0)
    mat = np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)
    return mat[:, :, :2].reshape(-1, 6).astype(np.float32)


def resolve_action_scale(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Read runtime scale from a rollout whose use rights were verified by its user."""
    if "action_scale" not in data.files:
        raise RuntimeError(
            "teacher rollout has no action_scale; public SONIC-A3 does not embed "
            "an external walking-policy contract"
        )
    joint_names = np.asarray(data["joint_names"]).astype(str)
    result = np.asarray(data["action_scale"], dtype=np.float32)
    if result.ndim == 2:
        if not np.allclose(result, result[:1], rtol=0.0, atol=1.0e-8):
            raise RuntimeError("teacher rollout changes action_scale across frames")
        result = result[0]
    if result.shape != (len(joint_names),):
        raise RuntimeError(
            f"action_scale shape {result.shape} does not match "
            f"{len(joint_names)} recorded joints"
        )
    if np.any(result <= 0.0) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid A3 action scales: {result}")
    return result


def build_clip(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    active_idx = np.flatnonzero(data["mode"].astype(str) == "ACTIVE")
    if len(active_idx) < 100:
        raise ValueError(f"{path}: insufficient ACTIVE samples")
    # The simulator records at 100 Hz; SONIC policy and teacher inference use 50 Hz.
    idx = active_idx[::2]
    q = data["joint_pos"][idx].astype(np.float32)
    dq = data["joint_vel"][idx].astype(np.float32)
    # The recorder stores physical target displacement (q_target-q_default),
    # while SONIC consumes normalized raw actions and applies
    # q_target = raw_action * action_scale + q_default.
    action_scale = resolve_action_scale(data)
    action = data["teacher_action"][idx].astype(np.float32) / action_scale
    default = data["default_joint_pos"].astype(np.float32)
    gravity = data["projected_gravity"][idx].astype(np.float32)
    ang = data["root_ang_vel_b"][idx].astype(np.float32)
    quat = data["root_quat_wxyz"][idx].astype(np.float32)
    hz = 50.0
    future_stride = round(DT_FUTURE * hz)

    q_rel = q - default
    previous_action = np.concatenate([action[:1] * 0.0, action[:-1]], axis=0)
    # Isaac Lab concatenates *terms after applying each term's history*.
    # PolicyCfg's runtime order is:
    #   base_ang_vel[10], joint_pos[10], joint_vel[10], actions[10], gravity_dir[10]
    # It is therefore term-major, not a flattened sequence of 93-D frames.
    # Keeping this contract exact is essential: a frame-major tensor has the
    # same 930-D shape and can achieve a tiny offline BC loss while being
    # meaningless to the decoder at runtime.
    if 3 + 29 + 29 + 29 + 3 != 93:
        raise RuntimeError("unexpected A3 proprioception frame width")

    encoder_inputs, proprio, targets = [], [], []
    last = len(q) - 1
    for i in range(len(q)):
        future_ids = np.minimum(i + np.arange(FUTURE) * future_stride, last)
        command_flat = np.concatenate([q[future_ids].reshape(-1), dq[future_ids].reshape(-1)])
        # Preserve the original observation's historical reshape quirk exactly:
        # [all future q, all future dq] -> [10, 58], then concatenate root 6D.
        command_nonflat = command_flat.reshape(FUTURE, 58)
        ori = relative_rot6d(quat[i], quat[future_ids])
        encoder_inputs.append(np.concatenate([command_nonflat, ori], axis=-1).reshape(-1))
        hist_ids = np.maximum(np.arange(i - HISTORY + 1, i + 1), 0)
        proprio.append(
            np.concatenate(
                [
                    ang[hist_ids].reshape(-1),
                    q_rel[hist_ids].reshape(-1),
                    dq[hist_ids].reshape(-1),
                    previous_action[hist_ids].reshape(-1),
                    gravity[hist_ids].reshape(-1),
                ]
            )
        )
        targets.append(action[i])
    return (
        np.asarray(encoder_inputs, dtype=np.float32),
        np.asarray(proprio, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def load_feature_clip(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    proprio_key = "proprio" if "proprio" in data.files else "native_proprio"
    missing = {"encoder_inputs", proprio_key, "targets"} - set(data.files)
    if missing:
        raise RuntimeError(f"{path}: missing feature arrays {sorted(missing)}")
    clip = (
        data["encoder_inputs"].astype(np.float32),
        data[proprio_key].astype(np.float32),
        data["targets"].astype(np.float32),
    )
    for array, expected in zip(clip, ((640,), (930,), (29,)), strict=True):
        if array.shape[1:] != expected:
            raise RuntimeError(f"{path}: expected (*, {expected}), got {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{path}: non-finite feature data")
    if len({len(array) for array in clip}) != 1:
        raise RuntimeError(f"{path}: feature arrays have different lengths")
    return clip


def load_schema2_feature_clip(
    path: Path,
    expected_action_contract_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load exact runtime token/proprio inputs from a schema-v2/v3 rollout."""
    with np.load(path, allow_pickle=False) as data:
        if "schema_json" not in data.files:
            raise RuntimeError(f"{path}: missing schema_json")
        metadata = json.loads(data["schema_json"].item())
        if metadata.get("schema_version") not in (2, 3):
            raise RuntimeError(
                f"{path}: exact Dynamic Decoder training requires schema 2 or 3, "
                f"got {metadata.get('schema_version')}"
            )
        actual_contract = metadata.get("runtime_action_contract_sha256")
        if actual_contract != expected_action_contract_sha256:
            raise RuntimeError(
                f"{path}: action contract mismatch: "
                f"dataset={actual_contract} expected={expected_action_contract_sha256}"
            )
        required = {
            "tokens",
            "native_proprio",
            "targets",
            "unique_motion_id",
            "done",
            "time_out",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"{path}: missing exact-rollout arrays {sorted(missing)}")
        tokens = data["tokens"].astype(np.float32)
        proprio = data["native_proprio"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        unique_motion_id = data["unique_motion_id"].astype(np.int64)
        done = data["done"].astype(bool)
        time_out = data["time_out"].astype(bool)

    arrays = (tokens, proprio, targets)
    for array, expected_shape in zip(arrays, ((64,), (930,), (29,)), strict=True):
        if array.shape[1:] != expected_shape:
            raise RuntimeError(
                f"{path}: expected (*, {expected_shape}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"{path}: non-finite feature data")
    if len({len(array) for array in arrays}) != 1:
        raise RuntimeError(f"{path}: feature arrays have different lengths")

    motion_keys = metadata.get("motion_keys")
    if (
        not isinstance(motion_keys, list)
        or not motion_keys
        or len(motion_keys) != len(set(motion_keys))
        or any(not isinstance(key, str) or not key for key in motion_keys)
    ):
        raise RuntimeError(f"{path}: invalid motion_keys metadata")
    if unique_motion_id.shape != (len(tokens),):
        raise RuntimeError(f"{path}: invalid unique_motion_id shape")
    if np.any(unique_motion_id < 0) or np.any(unique_motion_id >= len(motion_keys)):
        raise RuntimeError(f"{path}: unique_motion_id outside motion_keys")
    if done.shape != (len(tokens),) or time_out.shape != (len(tokens),):
        raise RuntimeError(f"{path}: invalid terminal-label shape")
    if np.any(time_out & ~done):
        raise RuntimeError(f"{path}: timeout without done")
    expected_episodes = int(metadata.get("num_envs", -1))
    terminal_count = int(np.count_nonzero(done))
    if terminal_count != expected_episodes:
        raise RuntimeError(
            f"{path}: incomplete episodes: terminal={terminal_count} "
            f"expected={expected_episodes}"
        )
    motion_names = np.asarray(motion_keys, dtype=str)[unique_motion_id]
    return tokens, proprio, targets, motion_names, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, action="append", default=[])
    parser.add_argument(
        "--feature-dataset",
        type=Path,
        action="append",
        default=[],
        help=(
            "NPZ captured inside SONIC with encoder_inputs[*,640], "
            "proprio/native_proprio[*,930], and normalized targets[*,29]."
        ),
    )
    parser.add_argument("--validation-rollout", type=Path, action="append", default=[])
    parser.add_argument(
        "--validation-feature-dataset",
        type=Path,
        action="append",
        default=[],
        help="Whole SONIC feature clips reserved for validation (no frame leakage).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--schema2-exact",
        action="store_true",
        help="use exact recorded 64-D runtime tokens from schema-v2/v3 rollouts",
    )
    parser.add_argument("--expected-action-contract-sha256")
    parser.add_argument("--balance-by-motion", action="store_true")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="validated fully composed config.yaml copied beside output checkpoints",
    )
    parser.add_argument(
        "--reset-dyn",
        action="store_true",
        help=(
            "Keep the checkpoint's A3 encoder but train the dynamic decoder "
            "from the deterministic PyTorch random initialization."
        ),
    )
    args = parser.parse_args()

    if not args.rollout and not args.feature_dataset:
        parser.error("at least one --rollout or --feature-dataset is required")
    if args.schema2_exact:
        if args.rollout or args.validation_rollout:
            parser.error("--schema2-exact does not accept legacy rollout inputs")
        if not args.validation_feature_dataset:
            parser.error("--schema2-exact requires a whole-file validation dataset")
        if not args.expected_action_contract_sha256:
            parser.error("--schema2-exact requires --expected-action-contract-sha256")
        if args.runtime_config is None:
            parser.error("--schema2-exact requires --runtime-config")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    schema2_train = []
    schema2_validation = []
    if args.schema2_exact:
        schema2_train = [
            load_schema2_feature_clip(
                path, args.expected_action_contract_sha256
            )
            for path in args.feature_dataset
        ]
        schema2_validation = [
            load_schema2_feature_clip(
                path, args.expected_action_contract_sha256
            )
            for path in args.validation_feature_dataset
        ]
        token_np = np.concatenate([clip[0] for clip in schema2_train])
        prop_np = np.concatenate([clip[1] for clip in schema2_train])
        target_np = np.concatenate([clip[2] for clip in schema2_train])
        train_motion_names = np.concatenate([clip[3] for clip in schema2_train])
        enc_np = None
    else:
        clips = [build_clip(path) for path in args.rollout]
        clips.extend(load_feature_clip(path) for path in args.feature_dataset)
        validation_clips = [build_clip(path) for path in args.validation_rollout]
        validation_clips.extend(
            load_feature_clip(path) for path in args.validation_feature_dataset
        )
        enc_np = np.concatenate([clip[0] for clip in clips])
        prop_np = np.concatenate([clip[1] for clip in clips])
        target_np = np.concatenate([clip[2] for clip in clips])
        if enc_np.shape[1:] != (640,):
            raise RuntimeError(f"encoder dataset shape mismatch: {enc_np.shape}")
        train_motion_names = None
    if prop_np.shape[1:] != (930,) or target_np.shape[1:] != (29,):
        raise RuntimeError(
            f"dataset shape mismatch: {prop_np.shape=} {target_np.shape=}"
        )
    if not all(np.isfinite(x).all() for x in (prop_np, target_np)):
        raise RuntimeError("non-finite teacher dataset")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy_state = checkpoint["policy_state_dict"]
    encoder = mlp([640, 2048, 1024, 512, 512, 64]).to(device)
    dyn = mlp([994, 2048, 2048, 1024, 1024, 512, 512, 29]).to(device)
    load_prefixed(encoder, policy_state, ENC_PREFIX)
    if not args.reset_dyn:
        load_prefixed(dyn, policy_state, DYN_PREFIX)
    encoder.eval().requires_grad_(False)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)

    def encode_features(
        clip_encoder: np.ndarray, clip_proprio: np.ndarray
    ) -> torch.Tensor:
        enc_tensor = torch.from_numpy(clip_encoder).to(device)
        token_parts = []
        with torch.inference_mode():
            for batch in enc_tensor.split(1024):
                latent = encoder(batch).reshape(-1, 2, 32)
                quantized, _ = quantizer(latent)
                token_parts.append(quantized.reshape(-1, 64).cpu())
        return torch.cat(
            [torch.cat(token_parts), torch.from_numpy(clip_proprio)], dim=-1
        )

    features = (
        torch.from_numpy(np.concatenate((token_np, prop_np), axis=1))
        if args.schema2_exact
        else encode_features(enc_np, prop_np)
    )
    targets = torch.from_numpy(target_np)

    if args.schema2_exact:
        val_tokens = np.concatenate([clip[0] for clip in schema2_validation])
        val_prop = np.concatenate([clip[1] for clip in schema2_validation])
        val_target = np.concatenate([clip[2] for clip in schema2_validation])
        train_ds = TensorDataset(features, targets)
        val_x = torch.from_numpy(np.concatenate((val_tokens, val_prop), axis=1)).to(device)
        val_y = torch.from_numpy(val_target).to(device)
        split_description = "whole_schema2_rollout_holdout"
    elif validation_clips:
        val_enc = np.concatenate([clip[0] for clip in validation_clips])
        val_prop = np.concatenate([clip[1] for clip in validation_clips])
        val_target = np.concatenate([clip[2] for clip in validation_clips])
        train_ds = TensorDataset(features, targets)
        val_x = encode_features(val_enc, val_prop).to(device)
        val_y = torch.from_numpy(val_target).to(device)
        split_description = "whole_clip_holdout"
    else:
        # Legacy fallback: deterministic random split. Adjacent motion frames
        # make this an optimistic metric, so do not use it for model selection.
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(features), generator=generator)
        split = max(1, int(0.9 * len(features)))
        train_ids, val_ids = order[:split], order[split:]
        train_ds = TensorDataset(features[train_ids], targets[train_ids])
        val_x, val_y = features[val_ids].to(device), targets[val_ids].to(device)
        split_description = "seeded_random_90_10"
    train_sampler = None
    if args.balance_by_motion:
        if train_motion_names is None:
            parser.error("--balance-by-motion requires --schema2-exact")
        motion_names, motion_counts = np.unique(train_motion_names, return_counts=True)
        count_by_motion = dict(
            zip(motion_names.tolist(), motion_counts.tolist(), strict=True)
        )
        weights = np.asarray(
            [1.0 / count_by_motion[name] for name in train_motion_names],
            dtype=np.float64,
        )
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(
            "A3_DYN_MOTION_BALANCE "
            + " ".join(
                f"{name}={int(count)}"
                for name, count in zip(motion_names, motion_counts, strict=True)
            ),
            flush=True,
        )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=1.0e-5)
    leg_indices = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], device=device)
    waist_indices = torch.tensor([2, 5, 8], device=device)
    locomotion_indices = torch.cat((leg_indices, waist_indices))
    arm_indices = torch.tensor(
        sorted(set(range(29)) - set(locomotion_indices.tolist())), device=device
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime_config is not None:
        if not args.runtime_config.is_file():
            raise RuntimeError(f"runtime config not found: {args.runtime_config}")
        shutil.copy2(args.runtime_config, args.output_dir / "config.yaml")
    metadata = {
        "rollouts": [str(path) for path in args.rollout],
        "feature_datasets": [str(path) for path in args.feature_dataset],
        "validation_rollouts": [str(path) for path in args.validation_rollout],
        "validation_feature_datasets": [
            str(path) for path in args.validation_feature_dataset
        ],
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "runtime_config": str(args.runtime_config) if args.runtime_config else None,
        "runtime_config_sha256": (
            file_sha256(args.runtime_config) if args.runtime_config else None
        ),
        "schema2_exact": args.schema2_exact,
        "expected_action_contract_sha256": args.expected_action_contract_sha256,
        "token_source": "recorded_runtime_tokens" if args.schema2_exact else "offline_encoder",
        "balance_by_motion": args.balance_by_motion,
        "train_motion_sample_counts": (
            {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(train_motion_names, return_counts=True), strict=True
                )
            }
            if train_motion_names is not None
            else None
        ),
        "dataset_sha256": {
            str(path): file_sha256(path)
            for path in args.feature_dataset + args.validation_feature_dataset
        },
        "samples": len(features),
        "train_samples": len(train_ds),
        "val_samples": len(val_x),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dyn_initialization": "random" if args.reset_dyn else "checkpoint",
        "seed": args.seed,
        "proprio_layout": "term_major:ang_vel,joint_pos,joint_vel,actions,gravity_dir",
        "split": split_description,
        "loss": "full_action_mse + 5 * locomotion_legs_and_waist_mse",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    writer = SummaryWriter(log_dir=args.output_dir / "runs")

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        full = (pred - target).square().mean()
        locomotion = (
            pred[:, locomotion_indices] - target[:, locomotion_indices]
        ).square().mean()
        return full + 5.0 * locomotion

    def save_checkpoint(path: Path, step: int) -> None:
        updated = checkpoint.copy()
        updated_state = policy_state.copy()
        for key, value in dyn.state_dict().items():
            updated_state[DYN_PREFIX + key] = value.detach().cpu()
        updated["policy_state_dict"] = updated_state
        updated["a3_dyn_bc_step"] = step
        updated["a3_dyn_bc_metadata"] = metadata
        torch.save(updated, path)

    best_val_loss = float("inf")

    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = dyn(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            with torch.inference_mode():
                val_pred = dyn(val_x)
                val_loss = loss_fn(val_pred, val_y)
                val_error_sq = (val_pred - val_y).square()
                val_rmse = val_error_sq.mean().sqrt()
                val_leg_rmse = val_error_sq[:, leg_indices].mean().sqrt()
                val_waist_rmse = val_error_sq[:, waist_indices].mean().sqrt()
                val_arm_rmse = val_error_sq[:, arm_indices].mean().sqrt()
            print(
                f"A3_DYN_BC step={step} train_loss={loss.item():.8f} "
                f"val_loss={val_loss.item():.8f} val_rmse={val_rmse.item():.8f} "
                f"leg_rmse={val_leg_rmse.item():.8f} "
                f"waist_rmse={val_waist_rmse.item():.8f} "
                f"arm_rmse={val_arm_rmse.item():.8f}",
                flush=True,
            )
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("loss/validation", val_loss.item(), step)
            writer.add_scalar("metrics/validation_applied_action_rmse", val_rmse.item(), step)
            writer.add_scalar("metrics/validation_leg_rmse", val_leg_rmse.item(), step)
            writer.add_scalar("metrics/validation_waist_rmse", val_waist_rmse.item(), step)
            writer.add_scalar("metrics/validation_arm_rmse", val_arm_rmse.item(), step)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                save_checkpoint(args.output_dir / "best_validation.pt", step)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(args.output_dir / f"a3_dyn_step_{step:06d}.pt", step)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
