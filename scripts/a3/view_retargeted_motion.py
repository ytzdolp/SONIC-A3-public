#!/usr/bin/env python3
"""Interactively replay retargeted A3 motion-library PKLs in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import mujoco
import mujoco.viewer
import numpy as np


def load_entries(path: Path) -> list[tuple[str, dict]]:
    files = sorted(path.rglob("*.pkl")) if path.is_dir() else [path]
    entries = []
    for file in files:
        entries.extend(joblib.load(file).items())
    if not entries:
        raise ValueError(f"No motion entries found under {path}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion", type=Path, default=Path("sample_data/a3_robot_filtered")
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    motions = load_entries(args.motion)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0
        while viewer.is_running():
            for name, motion in motions:
                fps = float(motion.get("fps", 30))
                root = np.asarray(motion["root_trans_offset"])
                quat_xyzw = np.asarray(motion["root_rot"])
                dof = np.asarray(motion["dof"])
                print(f"Playing {name}: {len(dof)} frames at {fps:g} fps", flush=True)
                for frame in range(len(dof)):
                    if not viewer.is_running():
                        return
                    start = time.perf_counter()
                    data.qpos[:3] = root[frame]
                    data.qpos[3:7] = quat_xyzw[frame, [3, 0, 1, 2]]
                    data.qpos[7:] = dof[frame]
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    remaining = 1.0 / (fps * args.speed) - (time.perf_counter() - start)
                    if remaining > 0:
                        time.sleep(remaining)


if __name__ == "__main__":
    main()
