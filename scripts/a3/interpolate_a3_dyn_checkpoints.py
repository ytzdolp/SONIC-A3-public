#!/usr/bin/env python3
"""Interpolate only the native A3 dynamics-decoder tensors of two checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DYN_PREFIX = "actor_module.decoders.a3_dyn.module."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=False)
    base_state = base["policy_state_dict"]
    tuned_state = tuned["policy_state_dict"]
    if base_state.keys() != tuned_state.keys():
        raise RuntimeError("checkpoint policy schemas differ")

    changed = []
    for key, base_value in base_state.items():
        tuned_value = tuned_state[key]
        if key.startswith(DYN_PREFIX):
            if base_value.shape != tuned_value.shape:
                raise RuntimeError(f"shape mismatch for {key}")
            if torch.is_floating_point(base_value):
                base_state[key] = torch.lerp(base_value, tuned_value, args.alpha)
            else:
                base_state[key] = tuned_value.clone()
            changed.append(key)
        elif not torch.equal(base_value, tuned_value):
            raise RuntimeError(f"non-dynamics tensor changed in tuned checkpoint: {key}")

    base["a3_dyn_interpolation"] = {
        "base": str(args.base),
        "tuned": str(args.tuned),
        "alpha": args.alpha,
        "tensor_count": len(changed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(
        f"A3_DYN_INTERPOLATED output={args.output} alpha={args.alpha:.3f} "
        f"tensors={len(changed)}"
    )


if __name__ == "__main__":
    main()
