#!/usr/bin/env python3
"""Measure the frozen G1/A3 encoder token gap on paired motion windows.

This is a controlled pre-training diagnostic, not a closed-loop evaluation.
Both encoders receive the same semantic 29-DOF trajectory and the same
root-relative orientation window.  A difference therefore comes from the
encoder weights, rather than joint ordering or source-motion mismatch.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch import nn
from vector_quantize_pytorch import FSQ


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal_assignment(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
                raise RuntimeError(f"{path}: {name} is not a literal integer list")
            return value
    raise RuntimeError(f"{path}: assignment {name} not found")


def load_checkpoint(path: Path) -> dict:
    # Older SONIC checkpoints pickle these TRL symbols at their former module path.
    import trl.trainer.utils as trainer_utils
    from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div

    trainer_utils.OnlineTrainerState = OnlineTrainerState
    trainer_utils.exact_div = exact_div
    return torch.load(path, map_location="cpu", weights_only=False)


def make_mlp(state: dict[str, torch.Tensor], prefix: str, device: torch.device) -> nn.Sequential:
    indices = sorted(
        int(key.removeprefix(prefix).split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    if not indices:
        raise RuntimeError(f"checkpoint contains no encoder tensors with prefix {prefix}")
    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        bias = state[f"{prefix}{index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layers.append(layer)
        if position != len(indices) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers).to(device).eval().requires_grad_(False)


def resample_linear(data: np.ndarray, source_fps: float, target_fps: int) -> np.ndarray:
    # MotionLib bypasses interpolation when the source already matches its
    # target FPS, so the final source frame must be retained in this case.
    if np.isclose(source_fps, target_fps, atol=1.0e-9, rtol=0.0):
        return np.asarray(data, dtype=np.float32).copy()
    duration = (len(data) - 1) / source_fps
    source_times = np.arange(len(data), dtype=np.float64) / source_fps
    target_times = np.arange(0.0, duration, 1.0 / target_fps, dtype=np.float64)
    result = np.empty((len(target_times), data.shape[1]), dtype=np.float32)
    for column in range(data.shape[1]):
        result[:, column] = np.interp(target_times, source_times, data[:, column])
    return result


def resample_quaternion_xyzw(
    data: np.ndarray, source_fps: float, target_fps: int
) -> np.ndarray:
    if np.isclose(source_fps, target_fps, atol=1.0e-9, rtol=0.0):
        return np.asarray(data, dtype=np.float32).copy()
    duration = (len(data) - 1) / source_fps
    source_times = np.arange(len(data), dtype=np.float64) / source_fps
    target_times = np.arange(0.0, duration, 1.0 / target_fps, dtype=np.float64)
    return Slerp(source_times, Rotation.from_quat(data))(target_times).as_quat().astype(np.float32)


def encoder_windows(
    entry: dict,
    mujoco_to_isaaclab: list[int],
    target_fps: int,
    num_future_frames: int,
    future_step: int,
) -> np.ndarray:
    source_fps = float(entry["fps"])
    dof = resample_linear(
        np.asarray(entry["dof"], dtype=np.float64)[:, mujoco_to_isaaclab],
        source_fps,
        target_fps,
    )
    quat = resample_quaternion_xyzw(
        np.asarray(entry["root_rot"], dtype=np.float64), source_fps, target_fps
    )
    velocity = np.diff(dof, axis=0) * target_fps
    # Match Humanoid_Batch.fk_batch: append the penultimate finite difference.
    velocity = np.concatenate([velocity, velocity[-2:-1]], axis=0)
    offsets = np.arange(num_future_frames) * future_step
    starts = np.arange(len(dof) - int(offsets[-1]))
    frame_indices = starts[:, None] + offsets[None, :]

    command = np.concatenate([dof[frame_indices], velocity[frame_indices]], axis=-1)
    base_rot = Rotation.from_quat(quat[starts])
    future_rot = Rotation.from_quat(quat[frame_indices.reshape(-1)])
    base_repeated = Rotation.from_quat(
        np.repeat(base_rot.as_quat(), num_future_frames, axis=0)
    )
    relative_matrix = (base_repeated.inv() * future_rot).as_matrix()
    root_six = relative_matrix[:, :2, :].reshape(len(starts), num_future_frames, 6)
    return np.concatenate([command, root_six.astype(np.float32)], axis=-1)


def cosine_summary(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    cosine = torch.nn.functional.cosine_similarity(first.flatten(1), second.flatten(1), dim=1)
    return {
        "mean": float(cosine.mean()),
        "p05": float(torch.quantile(cosine, 0.05)),
        "minimum": float(cosine.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-checkpoint", type=Path, required=True)
    parser.add_argument("--g1-robot-config", type=Path, default=Path("gear_sonic/envs/manager_env/robots/g1.py"))
    parser.add_argument("--a3-robot-config", type=Path, default=Path("gear_sonic/envs/manager_env/robots/a3.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    contract = manifest["contract"]
    g1_m2i = literal_assignment(args.g1_robot_config, "G1_MUJOCO_TO_ISAACLAB_DOF")
    a3_m2i = literal_assignment(args.a3_robot_config, "A3_MUJOCO_TO_ISAACLAB_DOF")
    g1_semantic_names = [contract["g1_mujoco_joint_names"][index] for index in g1_m2i]
    a3_semantic_names = [contract["a3_mujoco_joint_names"][index] for index in a3_m2i]
    if g1_semantic_names != a3_semantic_names:
        raise RuntimeError("G1/A3 Isaac Lab semantic DOF orders are not identical")

    ready_pairs = [pair for pair in manifest["pairs"] if pair["ready"]]
    if not ready_pairs:
        raise RuntimeError("manifest has no ready motion pairs")
    target_fps = int(contract["target_fps"])
    num_future_frames = int(contract["num_future_frames"])
    future_step = int(contract["future_step_at_target_fps"])

    paired_windows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for pair in ready_pairs:
        g1_entry = joblib.load(pair["g1_file"])[pair["motion_key"]]
        a3_entry = joblib.load(pair["a3_file"])[pair["motion_key"]]
        g1_windows = encoder_windows(g1_entry, g1_m2i, target_fps, num_future_frames, future_step)
        a3_windows = encoder_windows(a3_entry, a3_m2i, target_fps, num_future_frames, future_step)
        if g1_windows.shape != a3_windows.shape:
            raise RuntimeError(f"{pair['motion_key']}: paired window shapes differ")
        paired_windows.append((pair["motion_key"], g1_windows, a3_windows))

    g1_inputs_np = np.concatenate([item[1] for item in paired_windows])
    a3_inputs_np = np.concatenate([item[2] for item in paired_windows])
    expected_input_shape = (num_future_frames, 64)
    if g1_inputs_np.shape[1:] != expected_input_shape:
        raise RuntimeError(f"unexpected encoder window shape {g1_inputs_np.shape[1:]}")

    device = torch.device(args.device)
    g1_state = load_checkpoint(args.g1_checkpoint)["policy_state_dict"]
    a3_state = load_checkpoint(args.a3_checkpoint)["policy_state_dict"]
    g1_encoder = make_mlp(g1_state, "actor_module.encoders.g1.module.", device)
    a3_encoder = make_mlp(a3_state, "actor_module.encoders.a3.module.", device)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)

    g1_latents = []
    a3_latents = []
    g1_tokens = []
    a3_tokens = []
    with torch.inference_mode():
        for start in range(0, len(g1_inputs_np), args.batch_size):
            stop = min(start + args.batch_size, len(g1_inputs_np))
            g1_input = torch.from_numpy(g1_inputs_np[start:stop]).to(device).flatten(1)
            a3_input = torch.from_numpy(a3_inputs_np[start:stop]).to(device).flatten(1)
            g1_latent = g1_encoder(g1_input).reshape(-1, 2, 32)
            a3_latent = a3_encoder(a3_input).reshape(-1, 2, 32)
            g1_token, _ = quantizer(g1_latent)
            a3_token, _ = quantizer(a3_latent)
            g1_latents.append(g1_latent.cpu())
            a3_latents.append(a3_latent.cpu())
            g1_tokens.append(g1_token.cpu())
            a3_tokens.append(a3_token.cpu())

    g1_latent = torch.cat(g1_latents)
    a3_latent = torch.cat(a3_latents)
    g1_token = torch.cat(g1_tokens)
    a3_token = torch.cat(a3_tokens)
    latent_delta = g1_latent - a3_latent
    token_delta = g1_token - a3_token
    input_delta = g1_inputs_np - a3_inputs_np
    result = {
        "schema_version": 1,
        "diagnostic_scope": (
            "controlled paired-window encoder baseline; not closed-loop physics and not a generalization result"
        ),
        "manifest": str(args.manifest.resolve()),
        "g1_checkpoint": str(args.g1_checkpoint.resolve()),
        "a3_checkpoint": str(args.a3_checkpoint.resolve()),
        "g1_checkpoint_sha256": file_sha256(args.g1_checkpoint),
        "a3_checkpoint_sha256": file_sha256(args.a3_checkpoint),
        "device": str(device),
        "semantic_joint_names": g1_semantic_names,
        "motion_keys": [item[0] for item in paired_windows],
        "independent_source_motions": manifest["summary"]["independent_source_motions"],
        "windows": int(len(g1_inputs_np)),
        "input_contract": {
            "shape_per_window": list(expected_input_shape),
            "flattened_dim": int(np.prod(expected_input_shape)),
            "input_mae": float(np.abs(input_delta).mean()),
            "input_max_abs": float(np.abs(input_delta).max()),
        },
        "pre_quantization": {
            "shape": [2, 32],
            "rmse": float(torch.sqrt(torch.mean(latent_delta.square()))),
            "mae": float(latent_delta.abs().mean()),
            "cosine": cosine_summary(g1_latent, a3_latent),
        },
        "post_fsq": {
            "levels_per_dimension": 32,
            "shape": [2, 32],
            "rmse": float(torch.sqrt(torch.mean(token_delta.square()))),
            "mae": float(token_delta.abs().mean()),
            "cosine": cosine_summary(g1_token, a3_token),
            "element_exact_match_fraction": float(torch.mean((g1_token == a3_token).float())),
            "full_window_exact_match_fraction": float(
                torch.mean(torch.all(g1_token == a3_token, dim=(1, 2)).float())
            ),
        },
        "readiness": {
            "pipeline_smoke_test_ready": True,
            "alignment_training_ready": bool(manifest["summary"]["general_alignment_training_ready"]),
            "blocker": manifest["summary"]["general_alignment_blocker"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "G1_A3_TOKEN_BASELINE "
        f"windows={result['windows']} input_max={result['input_contract']['input_max_abs']:.3e} "
        f"latent_cos={result['pre_quantization']['cosine']['mean']:.6f} "
        f"token_cos={result['post_fsq']['cosine']['mean']:.6f} "
        f"token_exact={result['post_fsq']['element_exact_match_fraction']:.6f} "
        f"training_ready={result['readiness']['alignment_training_ready']}"
    )


if __name__ == "__main__":
    main()
