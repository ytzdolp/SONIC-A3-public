"""Summarize first termination per environment in A3 robustness evaluations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re


ROW = re.compile(
    r"TERMINATION_DIAG env=(?P<env>\d+) motion_frame=(?P<frame>\d+) "
    r"terms=(?P<terms>\[[^]]*\]).*?xy_error=(?P<xy>[0-9.]+).*?"
    r"yaw_error_deg=(?P<yaw>-?[0-9.]+).*?joint_error_max=(?P<joint>[0-9.]+)"
)


def read_first(path: Path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    rows = {}
    for match in ROW.finditer(text):
        env_id = int(match["env"])
        rows.setdefault(
            env_id,
            {
                "frame": int(match["frame"]),
                "terms": match["terms"],
                "xy": float(match["xy"]),
                "yaw": abs(float(match["yaw"])),
                "joint": float(match["joint"]),
            },
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="LABEL=PATH")
    args = parser.parse_args()

    datasets = {}
    for spec in args.logs:
        label, raw_path = spec.split("=", 1)
        rows = read_first(Path(raw_path))
        datasets[label] = rows
        outcomes = Counter(row["terms"] for row in rows.values())
        success = [row for row in rows.values() if row["terms"] == "['time_out']"]
        all_rows = list(rows.values())
        mean = lambda values: sum(values) / len(values) if values else float("nan")
        print(
            f"{label}: unique_envs={len(rows)} stable={len(success)}/{len(rows)} "
            f"outcomes={dict(outcomes)} frame_mean={mean([r['frame'] for r in all_rows]):.2f} "
            f"success_xy_mean={mean([r['xy'] for r in success]):.5f} "
            f"success_xy_max={max((r['xy'] for r in success), default=float('nan')):.5f} "
            f"success_yaw_mean={mean([r['yaw'] for r in success]):.3f} "
            f"success_yaw_max={max((r['yaw'] for r in success), default=float('nan')):.3f} "
            f"success_joint_mean={mean([r['joint'] for r in success]):.5f} "
            f"success_joint_max={max((r['joint'] for r in success), default=float('nan')):.5f}"
        )

    if len(datasets) == 2:
        (left_name, left), (right_name, right) = datasets.items()
        shared = sorted(left.keys() & right.keys())
        left_only = 0
        right_only = 0
        both = 0
        neither = 0
        for env_id in shared:
            left_ok = left[env_id]["terms"] == "['time_out']"
            right_ok = right[env_id]["terms"] == "['time_out']"
            both += left_ok and right_ok
            left_only += left_ok and not right_ok
            right_only += right_ok and not left_ok
            neither += not left_ok and not right_ok
        print(
            f"paired {left_name}->{right_name}: shared={len(shared)} both={both} "
            f"left_only={left_only} right_only={right_only} neither={neither}"
        )


if __name__ == "__main__":
    main()
