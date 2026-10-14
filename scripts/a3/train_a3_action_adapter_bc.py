#!/usr/bin/env python3
"""Train the A3 action adapter from exact runtime inputs.

Schema-v2 datasets store the tensor actually consumed by the adapter.  The
trainer refuses legacy or mixed-backbone data instead of reconstructing
runtime inputs from parallel observation fields.  Different adapter rounds
may be aggregated when their frozen policy backbone is identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from gear_sonic.utils.a3_adapter_contract import (
    ADAPTER_INPUT_DIM,
    ADAPTER_JOINT_INDICES,
    ADAPTER_PREFIX,
    NUM_ACTIONS,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
    base_policy_sha256,
    canonical_json_sha256,
    checkpoint_actor_state,
    file_sha256,
)
# Isaac evaluation uses high-precision TF32 GEMMs on small environment batches,
# while this standalone trainer validates in much larger batches.  The exact
# checkpoint SHA is authoritative; these bounds only allow the measured
# non-associative GEMM drift across batch shapes (runtime collection separately
# verifies the same-forward snapshot at 1e-6).
SOURCE_REPRODUCTION_MAX_TOLERANCE = 2.0e-3
SOURCE_REPRODUCTION_MEAN_TOLERANCE = 1.0e-4

def make_adapter(state: dict[str, torch.Tensor], device: torch.device) -> nn.Sequential:
    first = state.get(ADAPTER_PREFIX + "0.weight")
    last = state.get(ADAPTER_PREFIX + "2.weight")
    if first is None or last is None:
        raise RuntimeError("source checkpoint has no two-layer action residual adapter")
    if first.shape[1] != ADAPTER_INPUT_DIM or last.shape[0] != NUM_ACTIONS:
        raise RuntimeError(
            f"adapter contract mismatch: first={tuple(first.shape)} last={tuple(last.shape)} "
            f"expected_input={ADAPTER_INPUT_DIM} expected_output={NUM_ACTIONS}"
        )
    adapter = nn.Sequential(
        nn.Linear(first.shape[1], first.shape[0]),
        nn.SiLU(),
        nn.Linear(last.shape[1], last.shape[0]),
    ).to(device)
    selected = {
        key.removeprefix(ADAPTER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ADAPTER_PREFIX)
    }
    missing, unexpected = adapter.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"adapter state mismatch: missing={missing} unexpected={unexpected}")
    return adapter


def load_feature_file(
    path: Path,
    expected_base_policy_sha256: str,
    expected_max_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    if "schema_json" not in data.files:
        raise RuntimeError(f"{path}: missing schema_json")
    metadata = json.loads(data["schema_json"].item())
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: requires schema {DATASET_SCHEMA_VERSION}, "
            f"got {metadata.get('schema_version')}; legacy adapter data is unsafe"
        )
    actual_base_sha256 = metadata.get("student_base_policy_sha256")
    if actual_base_sha256 != expected_base_policy_sha256:
        raise RuntimeError(
            f"{path}: frozen backbone SHA mismatch: "
            f"dataset={actual_base_sha256} source={expected_base_policy_sha256}"
        )
    source_sha256 = metadata.get("student_checkpoint_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise RuntimeError(f"{path}: invalid student checkpoint SHA")
    action_contract_sha256 = metadata.get("runtime_action_contract_sha256")
    if not isinstance(action_contract_sha256, str) or len(action_contract_sha256) != 64:
        raise RuntimeError(f"{path}: invalid runtime action-contract SHA")
    action_contract = metadata.get("runtime_action_contract")
    if canonical_json_sha256(action_contract) != action_contract_sha256:
        raise RuntimeError(f"{path}: runtime action-contract payload/hash mismatch")
    if metadata.get("adapter_input_dim") != ADAPTER_INPUT_DIM:
        raise RuntimeError(f"{path}: adapter input-dimension metadata mismatch")
    recorded_indices = metadata.get("adapter_joint_indices")
    if (
        not isinstance(recorded_indices, list)
        or len(recorded_indices) != len(set(recorded_indices))
        or sorted(recorded_indices) != sorted(ADAPTER_JOINT_INDICES)
    ):
        raise RuntimeError(f"{path}: adapter joint-index metadata mismatch")
    recorded_max_abs = float(metadata.get("adapter_max_abs", float("nan")))
    if not np.isclose(recorded_max_abs, expected_max_abs, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(
            f"{path}: adapter max_abs mismatch: "
            f"dataset={recorded_max_abs} trainer={expected_max_abs}"
        )
    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != NUM_ACTIONS:
        raise RuntimeError(f"{path}: invalid 29-DoF joint-name metadata")
    selected_names = [joint_names[index] for index in ADAPTER_JOINT_INDICES]
    if any(not name for name in selected_names) or len(set(selected_names)) != len(
        selected_names
    ):
        raise RuntimeError(
            f"{path}: adapter indices do not select 15 unique A3 joints: "
            f"{selected_names}"
        )
    required = {
        "adapter_inputs",
        "student_base_actions",
        "student_adapter_residual",
        "raw_student_actions",
        "targets",
        "env_id",
        "global_step",
        "done",
        "time_out",
        "unique_motion_id",
    }
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"{path}: missing arrays {sorted(missing)}")
    features = data["adapter_inputs"].astype(np.float32)
    recorded_base = data["student_base_actions"].astype(np.float32)
    recorded_residual = data["student_adapter_residual"].astype(np.float32)
    raw_student = data["raw_student_actions"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    expected = (
        (ADAPTER_INPUT_DIM,),
        (NUM_ACTIONS,),
        (NUM_ACTIONS,),
        (NUM_ACTIONS,),
        (NUM_ACTIONS,),
    )
    arrays = (features, recorded_base, recorded_residual, raw_student, targets)
    for array, shape in zip(arrays, expected, strict=True):
        if array.shape[1:] != shape:
            raise RuntimeError(f"{path}: expected (*, {shape}), got {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{path}: non-finite values")
    if len({len(array) for array in arrays}) != 1:
        raise RuntimeError(f"{path}: inconsistent sample counts")
    if len(features) == 0:
        raise RuntimeError(f"{path}: empty adapter dataset")
    env_id = data["env_id"]
    global_step = data["global_step"]
    done = data["done"]
    time_out = data["time_out"]
    unique_motion_id = data["unique_motion_id"]
    for name, array in (
        ("env_id", env_id),
        ("global_step", global_step),
        ("done", done),
        ("time_out", time_out),
        ("unique_motion_id", unique_motion_id),
    ):
        if array.shape != (len(features),):
            raise RuntimeError(f"{path}: {name} must have shape ({len(features)},)")
    num_envs = metadata.get("num_envs")
    if not isinstance(num_envs, int) or num_envs <= 0:
        raise RuntimeError(f"{path}: missing or invalid num_envs metadata")
    expected_env_ids = np.arange(num_envs)
    actual_env_ids = np.unique(env_id)
    if not np.array_equal(actual_env_ids, expected_env_ids):
        raise RuntimeError(
            f"{path}: expected env ids 0..{num_envs - 1}, got {actual_env_ids.tolist()}"
        )
    if np.any(time_out & ~done):
        raise RuntimeError(f"{path}: time_out is true on a non-terminal sample")
    motion_keys = metadata.get("motion_keys")
    if (
        not isinstance(motion_keys, list)
        or not motion_keys
        or any(not isinstance(key, str) or not key for key in motion_keys)
        or len(motion_keys) != len(set(motion_keys))
    ):
        raise RuntimeError(f"{path}: missing or invalid motion_keys metadata")
    if np.any(unique_motion_id < 0) or np.any(unique_motion_id >= len(motion_keys)):
        raise RuntimeError(f"{path}: unique_motion_id is outside motion_keys")
    for current_env_id in expected_env_ids:
        rows = np.flatnonzero(env_id == current_env_id)
        terminal_rows = rows[done[rows]]
        if len(terminal_rows) != 1 or terminal_rows[0] != rows[-1]:
            raise RuntimeError(
                f"{path}: env {current_env_id} must have exactly one terminal "
                "sample and it must be the final recorded row"
            )
        if np.any(np.diff(global_step[rows]) <= 0):
            raise RuntimeError(
                f"{path}: env {current_env_id} has non-increasing global_step"
            )
    successful_episodes = int(np.count_nonzero(done & time_out))
    failed_episodes = int(np.count_nonzero(done & ~time_out))
    if successful_episodes != metadata.get("successful_episodes"):
        raise RuntimeError(f"{path}: successful episode count metadata mismatch")
    if failed_episodes != metadata.get("failed_episodes"):
        raise RuntimeError(f"{path}: failed episode count metadata mismatch")
    if successful_episodes + failed_episodes != num_envs:
        raise RuntimeError(f"{path}: incomplete episode collection")
    base_error = float(np.max(np.abs(features[:, :NUM_ACTIONS] - recorded_base)))
    action_error = float(
        np.max(np.abs(recorded_base + recorded_residual - raw_student))
    )
    if base_error > 1.0e-6 or action_error > 1.0e-5:
        raise RuntimeError(
            f"{path}: inconsistent recorded action contract: "
            f"base_error={base_error:.8g} action_error={action_error:.8g}"
        )
    return features, targets, recorded_residual, metadata


def validate_source_adapter(
    adapter: nn.Module,
    features: np.ndarray,
    recorded_residual: np.ndarray,
    mask: torch.Tensor,
    max_abs: float,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    max_error = 0.0
    absolute_sum = 0.0
    values = 0
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size]).to(device)
            expected = torch.from_numpy(
                recorded_residual[start : start + batch_size]
            ).to(device)
            actual = max_abs * torch.tanh(adapter(x)) * mask
            error = torch.abs(actual - expected)
            max_error = max(max_error, error.max().item())
            absolute_sum += error.sum().item()
            values += error.numel()
    mean_error = absolute_sum / values
    if (
        max_error > SOURCE_REPRODUCTION_MAX_TOLERANCE
        or mean_error > SOURCE_REPRODUCTION_MEAN_TOLERANCE
    ):
        raise RuntimeError(
            "source adapter does not reproduce recorded residuals: "
            f"max_error={max_error:.8g} mean_error={mean_error:.8g} "
            f"max_tolerance={SOURCE_REPRODUCTION_MAX_TOLERANCE:.8g} "
            f"mean_tolerance={SOURCE_REPRODUCTION_MEAN_TOLERANCE:.8g}"
        )
    return max_error, mean_error


def evaluate(
    adapter: nn.Module,
    loader: DataLoader,
    indices: torch.Tensor,
    max_abs: float,
    device: torch.device,
) -> dict[str, float]:
    squared_sum = 0.0
    absolute_sum = 0.0
    residual_absolute_sum = 0.0
    residual_absolute_max = 0.0
    values = 0
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            base_actions = features[:, :NUM_ACTIONS]
            residual = max_abs * torch.tanh(adapter(features))
            error = (
                base_actions.index_select(1, indices)
                + residual.index_select(1, indices)
                - targets.index_select(1, indices)
            )
            squared_sum += torch.square(error).sum().item()
            absolute_sum += torch.abs(error).sum().item()
            selected_residual = torch.abs(residual.index_select(1, indices))
            residual_absolute_sum += selected_residual.sum().item()
            residual_absolute_max = max(
                residual_absolute_max, selected_residual.max().item()
            )
            values += error.numel()
    return {
        "mse": squared_sum / values,
        "mae": absolute_sum / values,
        "residual_abs_mean": residual_absolute_sum / values,
        "residual_abs_max": residual_absolute_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dataset", type=Path, action="append", default=[])
    parser.add_argument(
        "--validation-feature-dataset", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--generalization-feature-dataset", type=Path, action="append", default=[]
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-abs", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balance-by-motion",
        action="store_true",
        help="sample each recorded motion id with equal probability",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.feature_dataset or not args.validation_feature_dataset:
        parser.error("both training and whole-file validation datasets are required")
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
        parser.error("steps, batch-size, and eval-every must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    train_paths = [path.resolve() for path in args.feature_dataset]
    val_paths = [path.resolve() for path in args.validation_feature_dataset]
    generalization_paths = [
        path.resolve() for path in args.generalization_feature_dataset
    ]
    overlap = set(train_paths) & (set(val_paths) | set(generalization_paths))
    overlap |= set(val_paths) & set(generalization_paths)
    if overlap:
        raise RuntimeError(
            "dataset files must be disjoint across train/validation/generalization: "
            f"{sorted(str(path) for path in overlap)}"
        )

    checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_key, actor_state = checkpoint_actor_state(checkpoint)
    source_base_policy_sha256 = base_policy_sha256(actor_state)
    adapter = make_adapter(actor_state, device)
    indices = torch.tensor(ADAPTER_JOINT_INDICES, dtype=torch.long, device=device)
    residual_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=device)
    residual_mask[indices] = 1.0

    train_clips = [
        load_feature_file(path, source_base_policy_sha256, args.max_abs)
        for path in args.feature_dataset
    ]
    val_clips = [
        load_feature_file(path, source_base_policy_sha256, args.max_abs)
        for path in args.validation_feature_dataset
    ]
    generalization_clips = [
        load_feature_file(path, source_base_policy_sha256, args.max_abs)
        for path in args.generalization_feature_dataset
    ]
    action_contract_hashes = {
        clip[3]["runtime_action_contract_sha256"]
        for clip in train_clips + val_clips + generalization_clips
    }
    if len(action_contract_hashes) != 1:
        raise RuntimeError(
            "mixed runtime action contracts are unsafe: "
            f"{sorted(action_contract_hashes)}"
        )
    runtime_action_contract_sha256 = next(iter(action_contract_hashes))
    train_x = np.concatenate([clip[0] for clip in train_clips])
    train_y = np.concatenate([clip[1] for clip in train_clips])
    train_motion_keys = np.concatenate(
        [
            np.asarray(clip[3]["motion_keys"], dtype=str)[
                np.load(path, allow_pickle=False)["unique_motion_id"]
            ]
            for path, clip in zip(args.feature_dataset, train_clips, strict=True)
        ]
    )
    val_x = np.concatenate([clip[0] for clip in val_clips])
    val_y = np.concatenate([clip[1] for clip in val_clips])
    generalization_x = (
        np.concatenate([clip[0] for clip in generalization_clips])
        if generalization_clips
        else None
    )
    generalization_y = (
        np.concatenate([clip[1] for clip in generalization_clips])
        if generalization_clips
        else None
    )

    def exact_source_samples(clips):
        exact = [
            clip
            for clip in clips
            if clip[3]["student_checkpoint_sha256"] == checkpoint_sha256
        ]
        if not exact:
            return None
        return (
            np.concatenate([clip[0] for clip in exact]),
            np.concatenate([clip[2] for clip in exact]),
        )

    exact_train = exact_source_samples(train_clips)
    exact_val = exact_source_samples(val_clips)
    exact_generalization = exact_source_samples(generalization_clips)
    if exact_train is None or exact_val is None:
        raise RuntimeError(
            "training and validation must each include schema-v2 data collected "
            "from the exact initialization checkpoint; older adapter rounds may "
            "be added only as same-backbone aggregation data"
        )

    train_sampler = None
    if args.balance_by_motion:
        motion_keys, motion_counts = np.unique(train_motion_keys, return_counts=True)
        count_by_motion = dict(zip(motion_keys.tolist(), motion_counts.tolist(), strict=True))
        sample_weights = np.asarray(
            [1.0 / count_by_motion[motion_key] for motion_key in train_motion_keys],
            dtype=np.float64,
        )
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(
            "A3_ADAPTER_MOTION_BALANCE "
            + " ".join(
                f"{motion_key}={int(count)}"
                for motion_key, count in zip(motion_keys, motion_counts, strict=True)
            ),
            flush=True,
        )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.validation_batch_size,
        shuffle=False,
        drop_last=False,
    )
    generalization_loader = (
        DataLoader(
            TensorDataset(
                torch.from_numpy(generalization_x),
                torch.from_numpy(generalization_y),
            ),
            batch_size=args.validation_batch_size,
            shuffle=False,
            drop_last=False,
        )
        if generalization_x is not None
        else None
    )
    source_train_max, source_train_mean = validate_source_adapter(
        adapter,
        exact_train[0],
        exact_train[1],
        residual_mask,
        args.max_abs,
        device,
        args.validation_batch_size,
    )
    source_val_max, source_val_mean = validate_source_adapter(
        adapter,
        exact_val[0],
        exact_val[1],
        residual_mask,
        args.max_abs,
        device,
        args.validation_batch_size,
    )
    source_generalization = (
        validate_source_adapter(
            adapter,
            exact_generalization[0],
            exact_generalization[1],
            residual_mask,
            args.max_abs,
            device,
            args.validation_batch_size,
        )
        if exact_generalization is not None
        else None
    )
    print(
        "A3_ADAPTER_SOURCE_CONTRACT "
        f"checkpoint_sha256={checkpoint_sha256} "
        f"train_max={source_train_max:.8g} train_mean={source_train_mean:.8g} "
        f"val_max={source_val_max:.8g} val_mean={source_val_mean:.8g}"
        + (
            f" generalization_max={source_generalization[0]:.8g} "
            f"generalization_mean={source_generalization[1]:.8g}"
            if source_generalization is not None
            else ""
        ),
        flush=True,
    )

    iterator = iter(train_loader)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config = args.checkpoint.parent / "config.yaml"
    if source_config.exists():
        shutil.copy2(source_config, args.output_dir / "config.yaml")
    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_base_policy_sha256": source_base_policy_sha256,
        "runtime_action_contract_sha256": runtime_action_contract_sha256,
        "dataset_student_checkpoint_sha256": sorted(
            {
                clip[3]["student_checkpoint_sha256"]
                for clip in train_clips + val_clips + generalization_clips
            }
        ),
        "feature_datasets": [str(path) for path in args.feature_dataset],
        "validation_feature_datasets": [
            str(path) for path in args.validation_feature_dataset
        ],
        "generalization_feature_datasets": [
            str(path) for path in args.generalization_feature_dataset
        ],
        "split": "whole_file_holdout",
        "train_samples": len(train_x),
        "validation_samples": len(val_x),
        "generalization_samples": (
            len(generalization_x) if generalization_x is not None else 0
        ),
        "adapter_input": "exact schema-v2 runtime adapter_inputs",
        "source_adapter_reproduction": {
            "train_max_error": source_train_max,
            "train_mean_error": source_train_mean,
            "validation_max_error": source_val_max,
            "validation_mean_error": source_val_mean,
            "generalization_max_error": (
                source_generalization[0]
                if source_generalization is not None
                else None
            ),
            "generalization_mean_error": (
                source_generalization[1]
                if source_generalization is not None
                else None
            ),
        },
        "adapter_joint_indices": list(ADAPTER_JOINT_INDICES),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_abs": args.max_abs,
        "grad_clip": args.grad_clip,
        "balance_by_motion": args.balance_by_motion,
        "train_motion_sample_counts": {
            str(motion_key): int(count)
            for motion_key, count in zip(
                *np.unique(train_motion_keys, return_counts=True), strict=True
            )
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    writer = SummaryWriter(log_dir=args.output_dir / "runs")

    def save_checkpoint(path: Path, step: int, validation: dict[str, float]) -> None:
        updated = checkpoint.copy()
        updated_state = actor_state.copy()
        for key, value in adapter.state_dict().items():
            updated_state[ADAPTER_PREFIX + key] = value.detach().cpu()
        updated[state_key] = updated_state
        updated["a3_action_adapter_bc_step"] = step
        updated["a3_action_adapter_bc_validation"] = validation
        updated["a3_action_adapter_bc_metadata"] = metadata
        torch.save(updated, path)

    initial_validation = evaluate(adapter, val_loader, indices, args.max_abs, device)
    initial_generalization = (
        evaluate(adapter, generalization_loader, indices, args.max_abs, device)
        if generalization_loader is not None
        else None
    )
    val_base = torch.from_numpy(val_x[:, :NUM_ACTIONS]).to(device)
    val_target = torch.from_numpy(val_y).to(device)
    with torch.inference_mode():
        delta = val_target.index_select(1, indices) - val_base.index_select(1, indices)
        oracle_error = torch.clamp(delta, -args.max_abs, args.max_abs) - delta
        oracle_bounded_mae = torch.abs(oracle_error).mean().item()
    print(
        "A3_ADAPTER_BC_BASELINE "
        f"val_mae={initial_validation['mae']:.8f} "
        f"val_mse={initial_validation['mse']:.8f} "
        f"bounded_oracle_mae={oracle_bounded_mae:.8f}"
        + (
            f" generalization_mae={initial_generalization['mae']:.8f}"
            if initial_generalization is not None
            else ""
        ),
        flush=True,
    )

    best_validation_mae = initial_validation["mae"]
    save_checkpoint(args.output_dir / "initial.pt", 0, initial_validation)
    for step in range(1, args.steps + 1):
        try:
            features, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            features, targets = next(iterator)
        features = features.to(device)
        targets = targets.to(device)
        base_actions = features[:, :NUM_ACTIONS]
        optimizer.zero_grad(set_to_none=True)
        residual = args.max_abs * torch.tanh(adapter(features))
        error = (
            base_actions.index_select(1, indices)
            + residual.index_select(1, indices)
            - targets.index_select(1, indices)
        )
        loss = torch.square(error).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(adapter, val_loader, indices, args.max_abs, device)
            generalization = (
                evaluate(
                    adapter,
                    generalization_loader,
                    indices,
                    args.max_abs,
                    device,
                )
                if generalization_loader is not None
                else None
            )
            train_mae = torch.abs(error).mean().item()
            print(
                f"A3_ADAPTER_BC step={step} train_mse={loss.item():.8f} "
                f"train_mae={train_mae:.8f} grad_norm={float(grad_norm):.8f} "
                f"val_mse={validation['mse']:.8f} val_mae={validation['mae']:.8f} "
                f"residual_mean={validation['residual_abs_mean']:.8f} "
                f"residual_max={validation['residual_abs_max']:.8f}"
                + (
                    f" generalization_mae={generalization['mae']:.8f}"
                    if generalization is not None
                    else ""
                ),
                flush=True,
            )
            writer.add_scalar("loss/train_mse", loss.item(), step)
            writer.add_scalar("metrics/train_mae", train_mae, step)
            writer.add_scalar("metrics/grad_norm", float(grad_norm), step)
            writer.add_scalar("loss/validation_mse", validation["mse"], step)
            writer.add_scalar("metrics/validation_mae", validation["mae"], step)
            writer.add_scalar(
                "policy/residual_abs_mean", validation["residual_abs_mean"], step
            )
            writer.add_scalar(
                "policy/residual_abs_max", validation["residual_abs_max"], step
            )
            if generalization is not None:
                writer.add_scalar(
                    "metrics/generalization_mae", generalization["mae"], step
                )
            if validation["mae"] < best_validation_mae:
                best_validation_mae = validation["mae"]
                save_checkpoint(args.output_dir / "best_validation.pt", step, validation)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                args.output_dir / f"adapter_step_{step:06d}.pt",
                step,
                validation if "validation" in locals() else initial_validation,
            )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
