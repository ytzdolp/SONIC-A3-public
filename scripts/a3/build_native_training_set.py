#!/usr/bin/env python3
"""Assemble validated A3 clips for native encoder/decoder training."""

import argparse
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "sample_data/a3_standing/a3_project_stand.pkl",
    ROOT / "sample_data/a3_robot_filtered/210531/walk_forward_amateur_001__A001.pkl",
    ROOT / "sample_data/a3_walking_curriculum/a3_step_in_place_pose050.pkl",
    ROOT / "sample_data/a3_walking_curriculum/a3_step_root015_pose050.pkl",
)
DEFAULT_OUTPUT = ROOT / "sample_data/a3_stage1_training/a3_stage1_train.pkl"
VIRTUAL_DIR = ROOT / "sample_data/a3_virtual"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input-dataset",
        type=Path,
        help="Filter an already combined dataset instead of reloading individual sources.",
    )
    parser.add_argument(
        "--include-virtual",
        action="store_true",
        help="Include numerically validated virtual clips for encoder/kinematic training.",
    )
    parser.add_argument(
        "--exclude-motion",
        action="append",
        default=[],
        help="Motion key to omit. Repeat for multiple keys.",
    )
    args = parser.parse_args()
    combined = {}
    if args.input_dataset is not None:
        sources = [args.input_dataset]
    else:
        sources = list(SOURCES)
        if args.include_virtual:
            sources.extend(sorted(VIRTUAL_DIR.glob("*.pkl")))
    for path in sources:
        motions = joblib.load(path)
        overlap = set(combined).intersection(motions)
        if overlap:
            raise RuntimeError(f"Duplicate motion keys: {sorted(overlap)}")
        for key, motion in motions.items():
            if key in args.exclude_motion:
                print(f"excluded motion={key} source={path}")
                continue
            if motion["dof"].shape[1] != 29:
                raise RuntimeError(f"{key}: expected 29 DOF")
            combined[key] = motion
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(combined, args.output)
    print(f"saved {args.output}")
    print(f"motions={len(combined)} frames={sum(len(m['dof']) for m in combined.values())}")


if __name__ == "__main__":
    main()
