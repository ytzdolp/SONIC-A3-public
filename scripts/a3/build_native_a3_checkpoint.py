#!/usr/bin/env python3
"""Create an independent A3 universal-token initialization from a G1 checkpoint.

G1 and A3 use different MuJoCo orderings, but their motion libraries convert to
the same name-verified Isaac Lab semantic order.  The network shapes therefore
permit an exact initialization while the resulting parameter namespace is A3-
specific and can subsequently be trained without modifying G1 modules.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


RENAMES = {
    "actor_module.encoders.g1.": "actor_module.encoders.a3.",
    "actor_module.decoders.g1_dyn.": "actor_module.decoders.a3_dyn.",
    "actor_module.decoders.g1_kin.": "actor_module.decoders.a3_kin.",
}


def rename_policy(state: dict[str, torch.Tensor]) -> OrderedDict:
    result = OrderedDict()
    for key, value in state.items():
        if key == "std":
            result[key] = value
            continue
        for source, target in RENAMES.items():
            if key.startswith(source):
                result[target + key[len(source) :]] = value
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = checkpoint["policy_state_dict"]
    a3_state = rename_policy(source_state)
    expected_prefixes = tuple(RENAMES.values())
    if not all(any(k.startswith(prefix) for k in a3_state) for prefix in expected_prefixes):
        raise RuntimeError("Source checkpoint is missing one or more G1 encoder/decoder modules")

    checkpoint["policy_state_dict"] = a3_state
    # Optimizer slots refer to the old module graph and must never be resumed.
    checkpoint["optimizer_state_dict"] = None
    checkpoint["lr_scheduler_state_dict"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved {args.output}")
    print(f"A3 policy tensors: {len(a3_state)}")
    for prefix in expected_prefixes:
        print(prefix, sum(k.startswith(prefix) for k in a3_state))


if __name__ == "__main__":
    main()
