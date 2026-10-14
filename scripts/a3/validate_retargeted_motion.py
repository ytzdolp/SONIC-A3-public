#!/usr/bin/env python3
"""Numerical quality checks for retargeted A3 motion-library files."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=Path("sample_data/a3_robot_filtered"))
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    limited = model.jnt_limited[1:].astype(bool)
    limits = model.jnt_range[1:]

    failed = False
    files = [args.motion] if args.motion.is_file() else sorted(args.motion.rglob("*.pkl"))
    if not files:
        raise SystemExit(f"No PKL motion files found under {args.motion}")
    for file in files:
        for name, motion in joblib.load(file).items():
            q = np.asarray(motion["dof"], dtype=np.float64)
            root = np.asarray(motion["root_trans_offset"], dtype=np.float64)
            quat = np.asarray(motion["root_rot"], dtype=np.float64)
            fps = float(motion["fps"])
            data = mujoco.MjData(model)
            foot_pos = np.empty((len(q), 2, 3))
            for frame in range(len(q)):
                data.qpos[:3] = root[frame]
                data.qpos[3:7] = quat[frame, [3, 0, 1, 2]]
                data.qpos[7:] = q[frame]
                mujoco.mj_forward(model, data)
                foot_pos[frame] = data.xpos[feet]

            below = np.maximum(limits[limited, 0] - q[:, limited], 0.0)
            above = np.maximum(q[:, limited] - limits[limited, 1], 0.0)
            violation = float(max(below.max(initial=0.0), above.max(initial=0.0)))
            joint_speed = np.abs(np.diff(q, axis=0)) * fps
            root_step = np.linalg.norm(np.diff(root, axis=0), axis=1)
            foot_speed = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=2) * fps
            contact = foot_pos[:-1, :, 2] < np.percentile(foot_pos[:, :, 2], 10) + 0.025
            contact_slip = foot_speed[contact]
            quat_norm_error = float(np.max(np.abs(np.linalg.norm(quat, axis=1) - 1.0)))
            print(
                f"{name}: frames={len(q)} limit_violation={violation:.6f}rad "
                f"max_joint_speed={joint_speed.max():.3f}rad/s "
                f"max_root_step={root_step.max():.4f}m "
                f"contact_slip_mean={contact_slip.mean():.3f}m/s "
                f"contact_slip_p95={np.percentile(contact_slip, 95):.3f}m/s "
                f"quat_norm_error={quat_norm_error:.2e}"
            )
            failed |= violation > 1.0e-5 or quat_norm_error > 1.0e-4 or root_step.max() > 0.1
    if failed:
        raise SystemExit("Retarget validation failed")


if __name__ == "__main__":
    main()
