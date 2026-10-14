#!/usr/bin/env python3
"""Evaluate G1→A3 latent adaptation through the frozen Round2 action stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from scripts.a3.audit_g1_a3_token_baseline import load_checkpoint, make_mlp
from scripts.a3.train_g1_a3_latent_adapter import LatentAdapter


NUM_ACTIONS = 29


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.square())))


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(first.flatten(1), second.flatten(1), dim=1).mean()
    )


def comparison(candidate: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    delta = candidate - native
    native_rms = rms(native)
    return {
        "mae": float(delta.abs().mean()),
        "rmse": rms(delta),
        "cosine": cosine(candidate, native),
        "relative_rmse_vs_native_rms": rms(delta) / max(native_rms, 1.0e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--a3-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    g1_state = load_checkpoint(args.g1_checkpoint)["policy_state_dict"]
    a3_checkpoint = load_checkpoint(args.a3_checkpoint)
    a3_state = a3_checkpoint["policy_state_dict"]
    adapter_checkpoint = torch.load(args.latent_adapter, map_location="cpu", weights_only=False)

    g1_encoder = make_mlp(g1_state, "actor_module.encoders.g1.module.", device)
    dynamic_decoder = make_mlp(a3_state, "actor_module.decoders.a3_dyn.module.", device)
    action_adapter = make_mlp(a3_state, "action_residual_adapter.", device)
    latent_adapter = LatentAdapter(adapter_checkpoint["hidden_dim"]).to(device).eval()
    latent_adapter.load_state_dict(adapter_checkpoint["state_dict"])
    latent_adapter.requires_grad_(False)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)

    with np.load(args.dataset, allow_pickle=False) as data:
        schema = json.loads(str(data["schema_json"].item()))
        arrays = {
            "encoder_inputs": data["encoder_inputs"].astype(np.float32),
            "native_tokens": data["tokens"].astype(np.float32),
            "proprio": data["native_proprio"].astype(np.float32),
            "adapter_inputs": data["adapter_inputs"].astype(np.float32),
            "targets": data["targets"].astype(np.float32),
            "motion_id": data["unique_motion_id"].astype(np.int64),
        }

    if arrays["encoder_inputs"].shape[1:] != (640,):
        raise RuntimeError("unexpected encoder input shape")
    if arrays["native_tokens"].shape[1:] != (64,):
        raise RuntimeError("unexpected token shape")
    if arrays["proprio"].shape[1:] != (930,):
        raise RuntimeError("unexpected proprioception shape")
    expected_adapter_dim = action_adapter[0].in_features
    if arrays["adapter_inputs"].shape[1:] != (expected_adapter_dim,):
        raise RuntimeError("unexpected action-adapter input shape")

    action_mask = torch.zeros(NUM_ACTIONS, device=device)
    action_mask[schema["adapter_joint_indices"]] = 1.0
    max_abs = float(schema["adapter_max_abs"])
    raw_clip = torch.tensor(schema["runtime_action_contract"]["raw_clip"], device=device)
    clip_low = raw_clip[:, 0]
    clip_high = raw_clip[:, 1]

    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "native_token",
            "raw_g1_token",
            "adapted_token",
            "native_base",
            "raw_g1_base",
            "adapted_base",
            "native_final",
            "raw_g1_final",
            "adapted_final",
            "native_clipped",
            "raw_g1_clipped",
            "adapted_clipped",
            "target",
            "motion_id",
        )
    }

    def decode(token: torch.Tensor, proprio: torch.Tensor, tail: torch.Tensor):
        base = dynamic_decoder(torch.cat([token.flatten(1), proprio], dim=-1))
        residual = max_abs * torch.tanh(
            action_adapter(torch.cat([base, tail], dim=-1))
        ) * action_mask
        final = base + residual
        clipped = torch.maximum(torch.minimum(final, clip_high), clip_low)
        return base, final, clipped

    with torch.inference_mode():
        for start in range(0, len(arrays["encoder_inputs"]), args.batch_size):
            stop = min(start + args.batch_size, len(arrays["encoder_inputs"]))
            encoder_input = torch.from_numpy(arrays["encoder_inputs"][start:stop]).to(device)
            native_token = torch.from_numpy(arrays["native_tokens"][start:stop]).to(device).reshape(-1, 2, 32)
            proprio = torch.from_numpy(arrays["proprio"][start:stop]).to(device)
            adapter_tail = torch.from_numpy(
                arrays["adapter_inputs"][start:stop, NUM_ACTIONS:]
            ).to(device)
            target = torch.from_numpy(arrays["targets"][start:stop]).to(device)

            g1_latent = g1_encoder(encoder_input).reshape(-1, 2, 32)
            raw_g1_token, _ = quantizer(g1_latent)
            adapted_latent = latent_adapter(g1_latent)
            adapted_token, _ = quantizer(adapted_latent)
            native_base, native_final, native_clipped = decode(native_token, proprio, adapter_tail)
            raw_g1_base, raw_g1_final, raw_g1_clipped = decode(
                raw_g1_token, proprio, adapter_tail
            )
            adapted_base, adapted_final, adapted_clipped = decode(
                adapted_token, proprio, adapter_tail
            )

            for name, value in (
                ("native_token", native_token),
                ("raw_g1_token", raw_g1_token),
                ("adapted_token", adapted_token),
                ("native_base", native_base),
                ("raw_g1_base", raw_g1_base),
                ("adapted_base", adapted_base),
                ("native_final", native_final),
                ("raw_g1_final", raw_g1_final),
                ("adapted_final", adapted_final),
                ("native_clipped", native_clipped),
                ("raw_g1_clipped", raw_g1_clipped),
                ("adapted_clipped", adapted_clipped),
                ("target", target),
            ):
                collected[name].append(value.cpu())
            collected["motion_id"].append(
                torch.from_numpy(arrays["motion_id"][start:stop]).long()
            )

    values = {name: torch.cat(parts) for name, parts in collected.items()}

    def result_for(mask: torch.Tensor) -> dict:
        result = {}
        for candidate in ("raw_g1", "adapted"):
            result[candidate] = {
                "token_vs_native": comparison(
                    values[f"{candidate}_token"][mask], values["native_token"][mask]
                ),
                "base_action_vs_native": comparison(
                    values[f"{candidate}_base"][mask], values["native_base"][mask]
                ),
                "final_action_vs_native": comparison(
                    values[f"{candidate}_final"][mask], values["native_final"][mask]
                ),
                "clipped_action_vs_native": comparison(
                    values[f"{candidate}_clipped"][mask], values["native_clipped"][mask]
                ),
                "clipped_target_mae": float(
                    (values[f"{candidate}_clipped"][mask] - values["target"][mask])
                    .abs()
                    .mean()
                ),
            }
        result["native_clipped_target_mae"] = float(
            (values["native_clipped"][mask] - values["target"][mask]).abs().mean()
        )
        return result

    all_mask = torch.ones(len(values["motion_id"]), dtype=torch.bool)
    per_motion = {}
    motion_keys = schema["motion_keys"]
    for motion_id in torch.unique(values["motion_id"]).tolist():
        key = motion_keys[motion_id] if 0 <= motion_id < len(motion_keys) else str(motion_id)
        mask = values["motion_id"] == motion_id
        per_motion[key] = {"samples": int(mask.sum()), **result_for(mask)}

    result = {
        "schema_version": 1,
        "scope": "offline frozen G1 encoder → latent adapter → FSQ → A3 dyn decoder + Round2 action adapter",
        "samples": len(values["motion_id"]),
        "dataset": str(args.dataset.resolve()),
        "g1_checkpoint": str(args.g1_checkpoint.resolve()),
        "a3_checkpoint": str(args.a3_checkpoint.resolve()),
        "latent_adapter": str(args.latent_adapter.resolve()),
        "overall": result_for(all_mask),
        "per_motion": per_motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
