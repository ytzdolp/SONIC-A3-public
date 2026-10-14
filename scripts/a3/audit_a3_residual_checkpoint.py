"""Audit that residual training changed only the bounded residual network."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


RESIDUAL_PREFIX = "actor_module.decoders.a3_dyn.residual_network."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)[
        "policy_state_dict"
    ]
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)[
        "policy_state_dict"
    ]
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        added = sorted(candidate.keys() - baseline.keys())
        raise RuntimeError(f"state keys differ: missing={missing}, added={added}")

    changed_residual = []
    changed_forbidden = []
    max_forbidden_error = 0.0
    for name in baseline:
        before = baseline[name]
        after = candidate[name]
        if not torch.is_tensor(before):
            if before != after:
                changed_forbidden.append(name)
            continue
        error = float((after - before).abs().max())
        if error == 0.0:
            continue
        if name.startswith(RESIDUAL_PREFIX):
            changed_residual.append((name, error))
        else:
            changed_forbidden.append(name)
            max_forbidden_error = max(max_forbidden_error, error)

    if changed_forbidden:
        raise RuntimeError(
            "non-residual actor tensors changed: "
            f"{changed_forbidden}; max_error={max_forbidden_error}"
        )
    if not changed_residual:
        raise RuntimeError("candidate did not change any residual tensor")

    limit_name = "actor_module.decoders.a3_dyn.residual_limit"
    limit = candidate[limit_name]
    if not torch.equal(limit, baseline[limit_name]):
        raise RuntimeError("residual_limit changed during training")

    print(f"baseline: {args.baseline}")
    print(f"candidate: {args.candidate}")
    print(f"changed_residual_tensors: {len(changed_residual)}")
    print(f"changed_non_residual_tensors: {len(changed_forbidden)}")
    print(f"max_non_residual_error: {max_forbidden_error:.9g}")
    print(f"residual_limit_min: {float(limit.min()):.6f}")
    print(f"residual_limit_max: {float(limit.max()):.6f}")
    print(
        "largest_residual_parameter_change: "
        f"{max(error for _, error in changed_residual):.9g}"
    )


if __name__ == "__main__":
    main()
