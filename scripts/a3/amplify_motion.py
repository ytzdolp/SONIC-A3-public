#!/usr/bin/env python3
"""Create a diagnostic motion by amplifying deviations from its first frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--suffix", default="amplified")
    args = parser.parse_args()
    if args.factor <= 0.0:
        raise ValueError("--factor must be positive")

    motions = joblib.load(args.input)
    amplified = {}
    for key, source in motions.items():
        motion = dict(source)
        # These fields are Euclidean or axis-angle trajectories. Amplifying
        # their deviation from frame zero preserves a continuous loop and the
        # original initial pose. Root quaternions are deliberately unchanged.
        for field in ("dof", "pose_aa", "root_trans_offset", "smpl_joints"):
            values = np.asarray(source[field])
            motion[field] = (
                values[0:1] + args.factor * (values - values[0:1])
            ).astype(values.dtype)
        amplified[f"{key}_{args.suffix}_{args.factor:g}x"] = motion

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(amplified, args.output)
    for key, motion in amplified.items():
        dof = np.asarray(motion["dof"])
        root = np.asarray(motion["root_trans_offset"])
        print(
            f"saved={args.output} motion={key} frames={len(dof)} "
            f"dof_ptp_max={np.ptp(dof, axis=0).max():.6f} "
            f"root_ptp={np.ptp(root, axis=0)}"
        )


if __name__ == "__main__":
    main()
