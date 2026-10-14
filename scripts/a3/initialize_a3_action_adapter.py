#!/usr/bin/env python3
"""Add the established zero-output A3 action adapter to a frozen checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from gear_sonic.utils.a3_adapter_contract import (
    ADAPTER_INPUT_DIM,
    ADAPTER_JOINT_INDICES,
    ADAPTER_PREFIX,
    NUM_ACTIONS,
    base_policy_sha256,
    checkpoint_actor_state,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state_key, source_state = checkpoint_actor_state(checkpoint)
    if any(key.startswith(ADAPTER_PREFIX) for key in source_state):
        raise RuntimeError("source checkpoint already contains an action adapter")

    torch.manual_seed(args.seed)
    adapter = nn.Sequential(
        nn.Linear(ADAPTER_INPUT_DIM, 128),
        nn.SiLU(),
        nn.Linear(128, NUM_ACTIONS),
    )
    nn.init.zeros_(adapter[2].weight)
    nn.init.zeros_(adapter[2].bias)

    updated_state = source_state.copy()
    for key, value in adapter.state_dict().items():
        updated_state[ADAPTER_PREFIX + key] = value.detach().cpu()
    checkpoint[state_key] = updated_state
    for key in ("optimizer_state_dict", "lr_scheduler_state_dict"):
        if key in checkpoint:
            checkpoint[key] = None
    checkpoint["a3_action_adapter_initialization"] = {
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": file_sha256(args.source),
        "source_base_policy_sha256": base_policy_sha256(source_state),
        "seed": args.seed,
        "input_dim": ADAPTER_INPUT_DIM,
        "hidden_dims": [128],
        "output_dim": NUM_ACTIONS,
        "max_abs": 0.25,
        "joint_indices": list(ADAPTER_JOINT_INDICES),
        "final_layer_zero_initialized": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output}")
    print(f"source_base_policy_sha256={base_policy_sha256(source_state)}")
    print(f"adapter_tensors={sum(k.startswith(ADAPTER_PREFIX) for k in updated_state)}")
    print(f"adapter_output_abs_max={adapter(torch.randn(4, ADAPTER_INPUT_DIM)).abs().max().item():.8g}")


if __name__ == "__main__":
    main()
