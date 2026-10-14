#!/usr/bin/env python3
"""Prepare G1/A3 alignment latents from schema-v2 closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from vector_quantize_pytorch import FSQ

from scripts.a3.audit_g1_a3_token_baseline import file_sha256, load_checkpoint, make_mlp


def load_feature_split(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    inputs = []
    motion_ids = []
    trajectories = []
    next_motion_id = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"encoder_inputs", "unique_motion_id", "tokens", "schema_json"}
            missing = sorted(required - set(data.files))
            if missing:
                raise RuntimeError(f"{path}: missing schema-v2 fields {missing}")
            encoder_input = data["encoder_inputs"].astype(np.float32)
            source_motion_id = data["unique_motion_id"].astype(np.int64)
            recorded_tokens = data["tokens"].astype(np.float32)
            schema = json.loads(str(data["schema_json"].item()))
        if encoder_input.shape[1:] != (640,) or recorded_tokens.shape[1:] != (64,):
            raise RuntimeError(f"{path}: unexpected encoder/token shape")
        if not np.isfinite(encoder_input).all() or not np.isfinite(recorded_tokens).all():
            raise RuntimeError(f"{path}: non-finite alignment features")
        remapped = np.empty(len(source_motion_id), dtype=np.int32)
        for source_id in np.unique(source_motion_id):
            mask = source_motion_id == source_id
            remapped[mask] = next_motion_id
            motion_keys = schema.get("motion_keys", [])
            key = (
                motion_keys[int(source_id)]
                if 0 <= int(source_id) < len(motion_keys)
                else f"motion_{int(source_id)}"
            )
            trajectories.append(
                {
                    "motion_id": next_motion_id,
                    "motion_key": key,
                    "source_motion_id": int(source_id),
                    "source_file": str(path.resolve()),
                    "source_sha256": file_sha256(path),
                    "samples": int(mask.sum()),
                    "successful_episodes": schema.get("successful_episodes"),
                    "failed_episodes": schema.get("failed_episodes"),
                }
            )
            next_motion_id += 1
        inputs.append(encoder_input)
        motion_ids.append(remapped)
    return np.concatenate(inputs), np.concatenate(motion_ids), trajectories


def encode(
    inputs: np.ndarray,
    g1_encoder: torch.nn.Module,
    a3_encoder: torch.nn.Module,
    quantizer: FSQ,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    result = {name: [] for name in ("g1_latent", "a3_latent", "g1_token", "a3_token")}
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
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
                result[name].append(value.cpu().numpy().astype(np.float32))
    return {name: np.concatenate(parts) for name, parts in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature", type=Path, action="append", required=True)
    parser.add_argument("--validation-feature", type=Path, action="append", required=True)
    parser.add_argument("--ood-dataset", type=Path)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_paths = [path.resolve() for path in args.train_feature]
    validation_paths = [path.resolve() for path in args.validation_feature]
    if set(train_paths) & set(validation_paths):
        raise RuntimeError("schema-v2 files must be disjoint across train and validation")

    device = torch.device(args.device)
    g1_state = load_checkpoint(args.g1_checkpoint)["policy_state_dict"]
    a3_state = load_checkpoint(args.a3_checkpoint)["policy_state_dict"]
    g1_encoder = make_mlp(g1_state, "actor_module.encoders.g1.module.", device)
    a3_encoder = make_mlp(a3_state, "actor_module.encoders.a3.module.", device)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "G1-to-A3 latent adapter on Round2 schema-v2 closed-loop states",
        "split_rule": "whole rollout file/seed holdout; no frame-level random split",
        "scope": "closed-loop locomotion commands represented in the Round2 datasets",
        "g1_checkpoint": str(args.g1_checkpoint.resolve()),
        "g1_checkpoint_sha256": file_sha256(args.g1_checkpoint),
        "a3_checkpoint": str(args.a3_checkpoint.resolve()),
        "a3_checkpoint_sha256": file_sha256(args.a3_checkpoint),
        "splits": {},
    }
    for split, paths in (("train", train_paths), ("validation", validation_paths)):
        inputs, motion_id, trajectories = load_feature_split(paths)
        encoded = encode(
            inputs, g1_encoder, a3_encoder, quantizer, device, args.batch_size
        )
        output_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(output_path, motion_id=motion_id, **encoded)
        manifest["splits"][split] = {
            "file": str(output_path.resolve()),
            "file_sha256": file_sha256(output_path),
            "samples": len(inputs),
            "trajectories": trajectories,
        }
        print(
            f"split={split} files={len(paths)} trajectories={len(trajectories)} "
            f"samples={len(inputs)} file={output_path}",
            flush=True,
        )

    if args.ood_dataset is not None:
        ood_output = args.output_dir / "ood.npz"
        shutil.copy2(args.ood_dataset, ood_output)
        manifest["splits"]["ood"] = {
            "file": str(ood_output.resolve()),
            "file_sha256": file_sha256(ood_output),
            "source_file": str(args.ood_dataset.resolve()),
            "source_sha256": file_sha256(args.ood_dataset),
            "role": "kinematic OOD evaluation only",
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
