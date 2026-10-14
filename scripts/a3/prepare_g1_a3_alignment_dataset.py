#!/usr/bin/env python3
"""Prepare a leakage-safe G1-to-A3 latent alignment dataset.

The source trajectories are A3 MotionLib files, but their 29-DOF semantic
Isaac Lab representation is also the input contract of the frozen G1 encoder.
No synthetic G1 physics trajectory is created: the same canonical window is
sent to both frozen encoders and their latent/token outputs are recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from vector_quantize_pytorch import FSQ

from scripts.a3.audit_g1_a3_token_baseline import (
    encoder_windows,
    file_sha256,
    literal_assignment,
    load_checkpoint,
    make_mlp,
)


TARGET_FPS = 50
NUM_FUTURE_FRAMES = 10
FUTURE_STEP = 5


def parse_source(spec: str) -> tuple[Path, str | None]:
    """Parse PATH or PATH::KEY_PREFIX without making ':' in paths ambiguous."""
    if "::" not in spec:
        return Path(spec), None
    path, prefix = spec.rsplit("::", 1)
    if not prefix:
        raise ValueError(f"empty key prefix in source specification: {spec}")
    return Path(path), prefix


def load_split(specs: list[str], a3_m2i: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    windows = []
    motion_ids = []
    motions = []
    seen_keys: set[str] = set()
    for spec in specs:
        path, prefix = parse_source(spec)
        payload = joblib.load(path)
        for key, entry in payload.items():
            if prefix is not None and not key.startswith(prefix):
                continue
            if key in seen_keys:
                raise RuntimeError(f"motion key occurs in more than one source: {key}")
            seen_keys.add(key)
            required = {"dof", "root_rot", "fps"}
            missing = sorted(required - set(entry))
            if missing:
                raise RuntimeError(f"{path}:{key}: missing fields {missing}")
            motion_windows = encoder_windows(
                entry,
                a3_m2i,
                TARGET_FPS,
                NUM_FUTURE_FRAMES,
                FUTURE_STEP,
            )
            if not np.isfinite(motion_windows).all():
                raise RuntimeError(f"{path}:{key}: non-finite encoder input")
            motion_index = len(motions)
            windows.append(motion_windows)
            motion_ids.append(np.full(len(motion_windows), motion_index, dtype=np.int32))
            motions.append(
                {
                    "motion_id": motion_index,
                    "motion_key": key,
                    "source_file": str(path.resolve()),
                    "source_sha256": file_sha256(path),
                    "source_frames": int(len(entry["dof"])),
                    "source_fps": float(entry["fps"]),
                    "alignment_windows": int(len(motion_windows)),
                }
            )
    if not windows:
        raise RuntimeError("source specifications selected no motions")
    return np.concatenate(windows), np.concatenate(motion_ids), motions


def encode(
    inputs: np.ndarray,
    g1_encoder: torch.nn.Module,
    a3_encoder: torch.nn.Module,
    quantizer: FSQ,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {
        "g1_latent": [],
        "a3_latent": [],
        "g1_token": [],
        "a3_token": [],
    }
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device).flatten(1)
            g1_latent = g1_encoder(batch).reshape(-1, 2, 32)
            a3_latent = a3_encoder(batch).reshape(-1, 2, 32)
            g1_token, _ = quantizer(g1_latent)
            a3_token, _ = quantizer(a3_latent)
            for name, value in (
                ("g1_latent", g1_latent),
                ("a3_latent", a3_latent),
                ("g1_token", g1_token),
                ("a3_token", a3_token),
            ):
                outputs[name].append(value.cpu().numpy().astype(np.float32))
    return {name: np.concatenate(values) for name, values in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", action="append", required=True)
    parser.add_argument("--validation-source", action="append", required=True)
    parser.add_argument("--ood-source", action="append", default=[])
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--a3-robot-config",
        type=Path,
        default=Path("gear_sonic/envs/manager_env/robots/a3.py"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    a3_m2i = literal_assignment(args.a3_robot_config, "A3_MUJOCO_TO_ISAACLAB_DOF")
    split_specs = {
        "train": args.train_source,
        "validation": args.validation_source,
    }
    if args.ood_source:
        split_specs["ood"] = args.ood_source

    loaded = {}
    all_motion_keys: dict[str, str] = {}
    for split, specs in split_specs.items():
        inputs, motion_id, motions = load_split(specs, a3_m2i)
        for motion in motions:
            key = motion["motion_key"]
            if key in all_motion_keys:
                raise RuntimeError(
                    f"motion-level leakage: {key} appears in {all_motion_keys[key]} and {split}"
                )
            all_motion_keys[key] = split
        loaded[split] = (inputs, motion_id, motions)

    device = torch.device(args.device)
    g1_state = load_checkpoint(args.g1_checkpoint)["policy_state_dict"]
    a3_state = load_checkpoint(args.a3_checkpoint)["policy_state_dict"]
    g1_encoder = make_mlp(g1_state, "actor_module.encoders.g1.module.", device)
    a3_encoder = make_mlp(a3_state, "actor_module.encoders.a3.module.", device)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "G1-to-A3 latent adapter pilot on shared semantic motion inputs",
        "scope": "locomotion pilot; not proof of universal cross-embodiment tokens",
        "contract": {
            "target_fps": TARGET_FPS,
            "num_future_frames": NUM_FUTURE_FRAMES,
            "future_step_frames": FUTURE_STEP,
            "features_per_frame": 64,
            "flattened_input_dim": 640,
            "latent_shape": [2, 32],
            "token_shape": [2, 32],
            "fsq_levels_per_dimension": 32,
        },
        "g1_checkpoint": str(args.g1_checkpoint.resolve()),
        "g1_checkpoint_sha256": file_sha256(args.g1_checkpoint),
        "a3_checkpoint": str(args.a3_checkpoint.resolve()),
        "a3_checkpoint_sha256": file_sha256(args.a3_checkpoint),
        "splits": {},
    }
    for split, (inputs, motion_id, motions) in loaded.items():
        encoded = encode(
            inputs,
            g1_encoder,
            a3_encoder,
            quantizer,
            device,
            args.batch_size,
        )
        output_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(
            output_path,
            encoder_input=inputs.astype(np.float32),
            motion_id=motion_id,
            **encoded,
        )
        manifest["splits"][split] = {
            "file": str(output_path.resolve()),
            "file_sha256": file_sha256(output_path),
            "motions": motions,
            "num_motions": len(motions),
            "num_windows": int(len(inputs)),
        }
        print(f"split={split} motions={len(motions)} windows={len(inputs)} file={output_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
