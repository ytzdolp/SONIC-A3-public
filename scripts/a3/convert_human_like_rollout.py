#!/usr/bin/env python3
"""Convert successful A3 human_like Isaac rollouts to SONIC MotionLib PKL."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np


def quaternion_xyzw_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to shortest-path axis angles."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-12)
    # q and -q encode the same rotation.  Choosing w >= 0 keeps the returned
    # angle in [0, pi] and avoids artificial 2*pi jumps in MotionLib.
    quaternion = np.where(quaternion[:, 3:4] < 0.0, -quaternion, quaternion)
    vector = quaternion[:, :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, quaternion[:, 3:4])
    scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1e-8)
    return (vector * scale).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    parser.add_argument("--source-hz", type=int, default=100)
    parser.add_argument("--output-hz", type=int, default=50)
    args = parser.parse_args()
    if args.source_hz % args.output_hz:
        raise ValueError("source-hz must be divisible by output-hz")
    stride = args.source_hz // args.output_hz

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    mj_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]
    if len(mj_names) != 29 or any(name is None for name in mj_names):
        raise RuntimeError(f"expected 29 named A3 hinge joints, got {mj_names}")

    output = {}
    for path in args.rollout:
        data = np.load(path)
        names = data["joint_names"].astype(str).tolist()
        if set(names) != set(mj_names):
            raise RuntimeError(
                f"{path}: schema mismatch missing={sorted(set(mj_names)-set(names))} "
                f"extra={sorted(set(names)-set(mj_names))}"
            )
        source_to_mj = np.asarray([names.index(name) for name in mj_names])
        active = np.flatnonzero(data["mode"].astype(str) == "ACTIVE")[::stride]
        if len(active) < 2:
            raise RuntimeError(f"{path}: expected at least two ACTIVE samples")
        dof = data["joint_pos"][active][:, source_to_mj].astype(np.float32)
        teacher_action = data["teacher_action"][active][:, source_to_mj].astype(np.float32)
        command = data["command"][active].astype(np.float32)
        raw_policy_action = data["raw_policy_action"][active].astype(np.float32)
        root = data["root_pos"][active].astype(np.float32)
        root[:, :2] -= root[0, :2]
        root_wxyz = data["root_quat_wxyz"][active].astype(np.float32)
        root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
        pose_aa = np.zeros((len(dof), 30, 3), dtype=np.float32)
        # MotionLib builds FK exclusively from pose_aa; the auxiliary root_rot
        # field is not consumed by load_motion_with_skeleton.  Root orientation
        # must therefore live in pose_aa[:, 0], otherwise every turning clip is
        # silently canonicalized to yaw=0.
        pose_aa[:, 0] = quaternion_xyzw_to_rotvec(root_xyzw)
        pose_aa[:, 1:] = model.jnt_axis[1:][None] * dof[:, :, None]
        key = path.stem.removeprefix("a3_teacher_").removesuffix("_20260810")
        output[f"a3_human_like_{key}"] = {
            "root_trans_offset": root,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_xyzw,
            "smpl_joints": np.zeros((len(dof), 24, 3), dtype=np.float32),
            "fps": args.output_hz,
            "teacher_action": teacher_action,
            # Keep the deployed velocity intent aligned with every reference
            # frame.  An external-policy observation can use this instead of
            # replaying one hard-coded forward command for every motion.
            "command": command,
            "raw_policy_action": raw_policy_action,
            "source_rollout": str(path),
        }
        print(
            f"motion=a3_human_like_{key} frames={len(dof)} "
            f"duration={len(dof)/args.output_hz:.2f}s"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output)
    print(f"saved={args.output} motions={len(output)} frames={sum(len(v['dof']) for v in output.values())}")


if __name__ == "__main__":
    main()
