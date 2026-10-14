#!/usr/bin/env python3
"""Retarget a SONIC G1 motion-library PKL to the 29-DOF Zhiyuan A3.

The two embodiments share semantic joint names.  Angles are mapped by name,
clamped to A3 limits, and the root-height offset is estimated with MuJoCo FK so
the feet retain the source motion's world height.  This is an initial
joint-space retarget suitable for bootstrapping tracking training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]


def foot_local_heights(model: mujoco.MjModel, dof: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    heights = np.empty((len(dof), 2), dtype=np.float32)
    for frame, angles in enumerate(dof):
        data.qpos[:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        data.qpos[7:] = angles
        mujoco.mj_forward(model, data)
        heights[frame] = data.xpos[feet, 2]
    return heights


def retarget_entry(entry: dict, g1: mujoco.MjModel, a3: mujoco.MjModel) -> tuple[dict, float]:
    source = np.asarray(entry["dof"], dtype=np.float64)
    g1_names = joint_names(g1)
    a3_names = joint_names(a3)
    if set(g1_names) != set(a3_names):
        missing = sorted(set(a3_names) - set(g1_names))
        extra = sorted(set(g1_names) - set(a3_names))
        raise ValueError(f"Joint-name mismatch: missing={missing}, extra={extra}")

    target = source[:, [g1_names.index(name) for name in a3_names]].copy()
    limited = a3.jnt_limited[1:].astype(bool)
    ranges = a3.jnt_range[1:]
    unclamped = target.copy()
    target[:, limited] = np.clip(
        target[:, limited], ranges[limited, 0] + 1.0e-4, ranges[limited, 1] - 1.0e-4
    )
    clamp_fraction = float(np.count_nonzero(abs(target - unclamped) > 1.0e-7) / target.size)

    root_trans = np.asarray(entry["root_trans_offset"], dtype=np.float32).copy()
    g1_foot_z = foot_local_heights(g1, source)
    a3_foot_z = foot_local_heights(a3, target)
    root_trans[:, 2] += float(np.median(g1_foot_z - a3_foot_z))

    root_rot = np.asarray(entry["root_rot"], dtype=np.float32).copy()
    pose_aa = np.zeros((len(target), 30, 3), dtype=np.float32)
    pose_aa[:, 0] = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa[:, 1:] = a3.jnt_axis[1:][None, :, :] * target[:, :, None]

    result = dict(entry)
    result.update(
        root_trans_offset=root_trans,
        pose_aa=pose_aa,
        dof=target.astype(np.float32),
        root_rot=root_rot,
    )
    return result, clamp_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--g1-mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"),
    )
    parser.add_argument(
        "--a3-mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    args = parser.parse_args()

    g1 = mujoco.MjModel.from_xml_path(str(args.g1_mjcf.resolve()))
    a3 = mujoco.MjModel.from_xml_path(str(args.a3_mjcf.resolve()))
    source = joblib.load(args.input)
    output = {}
    fractions = []
    for key, entry in source.items():
        output[key], fraction = retarget_entry(entry, g1, a3)
        fractions.append(fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output)
    print(
        f"Wrote {len(output)} motion(s) to {args.output}; "
        f"mean joint-limit clamp fraction={np.mean(fractions):.6f}"
    )


if __name__ == "__main__":
    main()
