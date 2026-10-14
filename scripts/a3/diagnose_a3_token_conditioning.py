#!/usr/bin/env python3
"""Measure how much an A3 Dynamic Decoder's final action depends on its token.

The test holds proprioception and the adapter's non-token observations fixed,
then compares the recorded token with a zero token and a token from another
motion.  It reconstructs both the frozen Dynamic Decoder and the residual
adapter directly from a checkpoint, so the result does not depend on physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


NUM_ACTIONS = 29
TOKEN_DIM = 64
PROPRIO_DIM = 930
ADAPTER_MAX_ABS = 0.25
ADAPTER_JOINT_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_mlp(state: dict[str, torch.Tensor], prefix: str, device: torch.device) -> nn.Sequential:
    layer_indices = sorted(
        int(key.removeprefix(prefix).split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    layers: list[nn.Module] = []
    for position, layer_index in enumerate(layer_indices):
        weight = state[f"{prefix}{layer_index}.weight"]
        bias = state[f"{prefix}{layer_index}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if position != len(layer_indices) - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers).to(device).eval()


def cross_motion_permutation(motion_ids: np.ndarray) -> np.ndarray:
    """Map every row to a deterministic row belonging to the next motion."""
    unique_ids = np.unique(motion_ids)
    if len(unique_ids) < 2:
        raise RuntimeError("token swap requires at least two unique motions")
    result = np.empty(len(motion_ids), dtype=np.int64)
    for index, motion_id in enumerate(unique_ids):
        source = np.flatnonzero(motion_ids == motion_id)
        target = np.flatnonzero(motion_ids == unique_ids[(index + 1) % len(unique_ids)])
        result[source] = target[np.arange(len(source)) % len(target)]
    if np.any(motion_ids[result] == motion_ids):
        raise RuntimeError("cross-motion permutation contains a same-motion token")
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["policy_state_dict"]
    dynamic_decoder = make_mlp(
        state, "actor_module.decoders.a3_dyn.module.", device
    )
    adapter = make_mlp(state, "action_residual_adapter.", device)

    with np.load(args.dataset, allow_pickle=False) as data:
        tokens_np = data["tokens"].astype(np.float32)
        proprio_np = data["native_proprio"].astype(np.float32)
        adapter_inputs_np = data["adapter_inputs"].astype(np.float32)
        recorded_base_np = data["student_base_actions"].astype(np.float32)
        recorded_residual_np = data["student_adapter_residual"].astype(np.float32)
        recorded_final_np = data["raw_student_actions"].astype(np.float32)
        targets_np = data["targets"].astype(np.float32)
        motion_ids = data["unique_motion_id"].astype(np.int64)

    expected_adapter_dim = adapter[0].in_features
    if tokens_np.shape[1:] != (TOKEN_DIM,) or proprio_np.shape[1:] != (PROPRIO_DIM,):
        raise RuntimeError("unexpected token/proprioception dimensions")
    if adapter_inputs_np.shape[1:] != (expected_adapter_dim,):
        raise RuntimeError("unexpected adapter input dimension")
    permutation = cross_motion_permutation(motion_ids)

    mask = torch.zeros(NUM_ACTIONS, device=device)
    mask[ADAPTER_JOINT_INDICES] = 1.0
    accumulators: dict[str, float] = {}
    sample_count = 0

    def add(name: str, value: torch.Tensor) -> None:
        accumulators[name] = accumulators.get(name, 0.0) + value.double().sum().item()

    with torch.inference_mode():
        for start in range(0, len(tokens_np), args.batch_size):
            stop = min(start + args.batch_size, len(tokens_np))
            tokens = torch.from_numpy(tokens_np[start:stop]).to(device)
            shuffled_tokens = torch.from_numpy(tokens_np[permutation[start:stop]]).to(device)
            proprio = torch.from_numpy(proprio_np[start:stop]).to(device)
            adapter_tail = torch.from_numpy(adapter_inputs_np[start:stop, NUM_ACTIONS:]).to(device)
            targets = torch.from_numpy(targets_np[start:stop]).to(device)

            recorded_base = torch.from_numpy(recorded_base_np[start:stop]).to(device)
            recorded_residual = torch.from_numpy(recorded_residual_np[start:stop]).to(device)
            recorded_final = torch.from_numpy(recorded_final_np[start:stop]).to(device)

            def forward(candidate_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                base = dynamic_decoder(torch.cat([candidate_tokens, proprio], dim=-1))
                residual = ADAPTER_MAX_ABS * torch.tanh(
                    adapter(torch.cat([base, adapter_tail], dim=-1))
                ) * mask
                return base, base + residual

            base, final = forward(tokens)
            zero_base, zero_final = forward(torch.zeros_like(tokens))
            shuffled_base, shuffled_final = forward(shuffled_tokens)

            add("base_reconstruction_abs", (base - recorded_base).abs())
            add("final_reconstruction_abs", (final - recorded_final).abs())
            reconstructed_residual = final - base
            add("residual_reconstruction_abs", (reconstructed_residual - recorded_residual).abs())
            add("original_target_abs", (final - targets).abs())
            add("zero_target_abs", (zero_final - targets).abs())
            add("shuffled_target_abs", (shuffled_final - targets).abs())
            add("zero_base_change_sq", (zero_base - base).square())
            add("zero_final_change_sq", (zero_final - final).square())
            add("shuffled_base_change_sq", (shuffled_base - base).square())
            add("shuffled_final_change_sq", (shuffled_final - final).square())
            add("original_final_sq", final.square())
            sample_count += stop - start

    scalar_count = sample_count * NUM_ACTIONS
    original_rms = (accumulators["original_final_sq"] / scalar_count) ** 0.5

    def rms(name: str) -> float:
        return (accumulators[name] / scalar_count) ** 0.5

    zero_base_rms = rms("zero_base_change_sq")
    zero_final_rms = rms("zero_final_change_sq")
    shuffled_base_rms = rms("shuffled_base_change_sq")
    shuffled_final_rms = rms("shuffled_final_change_sq")
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "samples": sample_count,
        "unique_motions": int(len(np.unique(motion_ids))),
        "device": str(device),
        "contract_checks": {
            "base_reconstruction_mae": accumulators["base_reconstruction_abs"] / scalar_count,
            "residual_reconstruction_mae": accumulators["residual_reconstruction_abs"] / scalar_count,
            "final_reconstruction_mae": accumulators["final_reconstruction_abs"] / scalar_count,
        },
        "target_mae": {
            "original_token": accumulators["original_target_abs"] / scalar_count,
            "zero_token": accumulators["zero_target_abs"] / scalar_count,
            "cross_motion_token": accumulators["shuffled_target_abs"] / scalar_count,
        },
        "action_change_rms": {
            "zero_token_base": zero_base_rms,
            "zero_token_final": zero_final_rms,
            "cross_motion_token_base": shuffled_base_rms,
            "cross_motion_token_final": shuffled_final_rms,
            "original_final_action": original_rms,
        },
        "relative_final_change": {
            "zero_token_vs_action_rms": zero_final_rms / original_rms,
            "cross_motion_token_vs_action_rms": shuffled_final_rms / original_rms,
        },
        "adapter_change_retention": {
            "zero_token_final_over_base": zero_final_rms / zero_base_rms,
            "cross_motion_token_final_over_base": shuffled_final_rms / shuffled_base_rms,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
