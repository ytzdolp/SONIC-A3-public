#!/usr/bin/env python3
"""Generate conservative, smooth A3 motions from the project robot contract.

These references are deliberately double-support motions.  They expand the
native encoder/kinematic training distribution without pretending to be a
walking teacher.  Every generated clip must still pass floating-base Isaac Lab
validation before being admitted to a training set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco
import numpy as np

from make_standing_motion import STAND_BY_JOINT


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]


def smooth_signed(t: np.ndarray, duration: float, cycles: float = 2.0) -> np.ndarray:
    envelope = np.sin(np.pi * t / duration) ** 2
    return envelope * np.sin(2.0 * np.pi * cycles * t / duration)


def smooth_positive(t: np.ndarray, duration: float, cycles: float = 2.0) -> np.ndarray:
    envelope = np.sin(np.pi * t / duration) ** 2
    return envelope * 0.5 * (1.0 - np.cos(2.0 * np.pi * cycles * t / duration))


def preserve_foot_height(
    model: mujoco.MjModel, dof: np.ndarray, base_root_height: float
) -> np.ndarray:
    """Adjust reference root z so mean ankle-link height matches frame zero."""
    data = mujoco.MjData(model)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    local_height = np.empty(len(dof), dtype=np.float64)
    for frame, q in enumerate(dof):
        data.qpos[:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        local_height[frame] = data.xpos[feet, 2].mean()
    return base_root_height + local_height[0] - local_height


def apply(dof: np.ndarray, index: dict[str, int], name: str, value: np.ndarray) -> None:
    dof[:, index[name]] += value


def build_clips(
    model: mujoco.MjModel, seconds: float, fps: int, root_height: float
) -> dict[str, dict]:
    names = joint_names(model)
    index = {name: i for i, name in enumerate(names)}
    frames = round(seconds * fps) + 1
    t = np.linspace(0.0, seconds, frames, dtype=np.float64)
    stand = np.asarray([STAND_BY_JOINT.get(name, 0.0) for name in names], dtype=np.float64)

    clips: dict[str, np.ndarray] = {}

    q = np.repeat(stand[None], frames, axis=0)
    squat = smooth_positive(t, seconds, cycles=2.0)
    for side in ("left", "right"):
        apply(q, index, f"{side}_hip_pitch_joint", -0.10 * squat)
        apply(q, index, f"{side}_knee_joint", 0.20 * squat)
        apply(q, index, f"{side}_ankle_pitch_joint", -0.10 * squat)
    clips["a3_virtual_shallow_squat"] = q

    q = np.repeat(stand[None], frames, axis=0)
    lateral = smooth_signed(t, seconds, cycles=2.0)
    # Small coordinated ankle/hip/waist roll motion; both feet remain planted.
    for side in ("left", "right"):
        apply(q, index, f"{side}_ankle_roll_joint", 0.025 * lateral)
        apply(q, index, f"{side}_hip_roll_joint", 0.035 * lateral)
    apply(q, index, "waist_roll_joint", -0.025 * lateral)
    clips["a3_virtual_lateral_weight_shift"] = q

    q = np.repeat(stand[None], frames, axis=0)
    twist = smooth_signed(t, seconds, cycles=2.0)
    apply(q, index, "waist_yaw_joint", 0.18 * twist)
    apply(q, index, "left_shoulder_yaw_joint", -0.12 * twist)
    apply(q, index, "right_shoulder_yaw_joint", -0.12 * twist)
    clips["a3_virtual_waist_twist"] = q

    q = np.repeat(stand[None], frames, axis=0)
    arms = smooth_signed(t, seconds, cycles=2.0)
    apply(q, index, "left_shoulder_pitch_joint", 0.28 * arms)
    apply(q, index, "right_shoulder_pitch_joint", -0.28 * arms)
    apply(q, index, "left_elbow_joint", 0.12 * arms)
    apply(q, index, "right_elbow_joint", -0.12 * arms)
    clips["a3_virtual_opposed_arm_swing"] = q

    q = np.repeat(stand[None], frames, axis=0)
    reach = smooth_positive(t, seconds, cycles=2.0)
    apply(q, index, "left_shoulder_pitch_joint", -0.35 * reach)
    apply(q, index, "right_shoulder_pitch_joint", -0.35 * reach)
    apply(q, index, "left_elbow_joint", -0.20 * reach)
    apply(q, index, "right_elbow_joint", -0.20 * reach)
    apply(q, index, "waist_pitch_joint", -0.04 * reach)
    clips["a3_virtual_bilateral_reach"] = q

    output: dict[str, dict] = {}
    limited = model.jnt_limited[1:].astype(bool)
    limits = model.jnt_range[1:]
    for motion_name, dof in clips.items():
        if np.any(dof[:, limited] < limits[limited, 0]) or np.any(
            dof[:, limited] > limits[limited, 1]
        ):
            raise ValueError(f"{motion_name} violates A3 joint limits")
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 2] = preserve_foot_height(model, dof, root_height).astype(np.float32)
        quat = np.repeat(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), frames, axis=0
        )
        pose_aa = np.zeros((frames, 30, 3), dtype=np.float32)
        pose_aa[:, 1:] = model.jnt_axis[1:][None] * dof[:, :, None]
        output[motion_name] = {
            "root_trans_offset": root,
            "pose_aa": pose_aa,
            "dof": dof.astype(np.float32),
            "root_rot": quat,
            "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
            "fps": fps,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("gear_sonic/data/assets/robot_description/mjcf/a3.xml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("sample_data/a3_virtual")
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--root-height", type=float, default=1.063)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    clips = build_clips(model, args.seconds, args.fps, args.root_height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in clips.items():
        path = args.output_dir / f"{name}.pkl"
        joblib.dump({name: entry}, path)
        print(
            f"saved={path} frames={len(entry['dof'])} fps={entry['fps']} "
            f"root_z=[{entry['root_trans_offset'][:, 2].min():.4f},"
            f"{entry['root_trans_offset'][:, 2].max():.4f}]"
        )


if __name__ == "__main__":
    main()
