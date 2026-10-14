#!/usr/bin/env python3
"""Generate a stationary A3 motion-library clip from the project stand pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np


STAND_BY_JOINT = {
    "left_hip_pitch_joint": -0.1311,
    "right_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "right_hip_roll_joint": -0.0056,
    "left_hip_yaw_joint": -0.0348,
    "right_hip_yaw_joint": 0.0348,
    "left_knee_joint": 0.2468,
    "right_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "right_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_ankle_roll_joint": 0.0078,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
    "left_elbow_joint": 0.8,
    "right_elbow_joint": 0.8,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]
    pose = np.array([STAND_BY_JOINT.get(name, 0.0) for name in names], dtype=np.float32)
    frames = round(args.seconds * args.fps)
    dof = np.repeat(pose[None], frames, axis=0)
    root = np.repeat(np.array([[0.0, 0.0, 1.063]], dtype=np.float32), frames, axis=0)
    quat = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), frames, axis=0)
    pose_aa = np.zeros((frames, 30, 3), dtype=np.float32)
    pose_aa[:, 1:] = model.jnt_axis[1:][None] * dof[:, :, None]
    entry = {
        "a3_project_stand": {
            "root_trans_offset": root,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": quat,
            "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
            "fps": args.fps,
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(entry, args.output)
    print(f"Wrote {frames} frames of project A3 standing to {args.output}")


if __name__ == "__main__":
    main()
