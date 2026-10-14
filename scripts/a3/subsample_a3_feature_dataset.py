#!/usr/bin/env python3
"""Deterministically thin an A3 feature dataset while preserving trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")

    source = np.load(args.input, allow_pickle=False)
    count = len(source["targets"])
    if "env_id" not in source.files:
        raise RuntimeError("dataset has no env_id trajectory field")
    keep_parts = []
    for env_id in np.unique(source["env_id"]):
        rows = np.flatnonzero(source["env_id"] == env_id)
        keep_parts.append(rows[:: args.stride])
        # Always preserve terminal-labelled frames when present.
        if "done" in source.files:
            keep_parts.append(rows[source["done"][rows]])
    keep = np.unique(np.concatenate(keep_parts))

    output = {}
    for key in source.files:
        value = source[key]
        if value.ndim > 0 and len(value) == count:
            output[key] = value[keep]
        elif key != "schema_json":
            output[key] = value
    metadata = (
        json.loads(str(source["schema_json"])) if "schema_json" in source.files else {}
    )
    metadata.update(
        {
            "source": str(args.input),
            "source_samples": count,
            "samples": len(keep),
            "subsample_stride": args.stride,
        }
    )
    output["schema_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(
        f"A3_SUBSAMPLE_SAVED path={args.output} source={count} "
        f"samples={len(keep)} stride={args.stride}"
    )


if __name__ == "__main__":
    main()
