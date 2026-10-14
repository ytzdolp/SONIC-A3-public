#!/usr/bin/env python3
"""Build a small-amplitude A3 walking curriculum from stand and full walk clips."""

from pathlib import Path
import argparse

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
STAND = ROOT / "sample_data/a3_standing/a3_project_stand.pkl"
WALK = ROOT / "sample_data/a3_robot_filtered/210531/walk_forward_amateur_001__A001.pkl"
TRANSITION_SECONDS = 2.0
# Six-second window with the largest net forward displacement in the source
# clip (about 4.04 m before curriculum scaling).
WALK_START = 911
WALK_END = 1091


def load_one(path: Path) -> dict:
    return next(iter(joblib.load(path).values()))


def aligned_quaternions(
    stand_quat: np.ndarray, walk_quat: np.ndarray, scale: float
) -> np.ndarray:
    """Align the complete walking orientation trajectory to the standing heading."""
    stand_r = Rotation.from_quat(stand_quat)
    walk_r = Rotation.from_quat(walk_quat)
    correction = stand_r * walk_r[0].inv()
    aligned = correction * walk_r

    # Keep only SCALE of the orientation excursion around the standing frame.
    relative = stand_r.inv() * aligned
    keys = Rotation.concatenate([Rotation.identity(), relative])
    result = []
    for rel in relative:
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([Rotation.identity(), rel]))
        result.append((stand_r * slerp([scale])[0]).as_quat())
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--root-scale", type=float, default=None)
    parser.add_argument("--pose-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root_scale = args.scale if args.root_scale is None else args.root_scale
    pose_scale = args.scale if args.pose_scale is None else args.pose_scale
    scale_tag = f"{round(root_scale * 100):03d}"
    output_path = args.output or (
        ROOT / f"sample_data/a3_walking_curriculum/a3_easy_step_{scale_tag}.pkl"
    )
    stand = load_one(STAND)
    walk = load_one(WALK)
    fps = int(walk["fps"])
    n_transition = int(TRANSITION_SECONDS * fps)
    stand_dof = np.asarray(stand["dof"][0], dtype=np.float32)
    stand_pose = np.asarray(stand["pose_aa"][0], dtype=np.float32)
    stand_smpl = np.asarray(stand["smpl_joints"][0], dtype=np.float32)
    stand_root = np.asarray(stand["root_trans_offset"][0], dtype=np.float32)
    stand_quat = np.asarray(stand["root_rot"][0], dtype=np.float32)

    walk_dof = np.asarray(walk["dof"][WALK_START:WALK_END], dtype=np.float32)
    walk_pose = np.asarray(walk["pose_aa"][WALK_START:WALK_END], dtype=np.float32)
    walk_smpl = np.asarray(walk["smpl_joints"][WALK_START:WALK_END], dtype=np.float32)
    walk_root = np.asarray(walk["root_trans_offset"][WALK_START:WALK_END], dtype=np.float32)
    walk_quat = np.asarray(walk["root_rot"][WALK_START:WALK_END], dtype=np.float32)

    # Scale motion around its first frame, while anchoring the pose to the exact
    # project A3 standing configuration learned by the warm-start policy.
    dof = stand_dof + pose_scale * (walk_dof - walk_dof[0])
    pose = stand_pose + pose_scale * (walk_pose - walk_pose[0])
    smpl = stand_smpl + pose_scale * (walk_smpl - walk_smpl[0])
    # The source human walks along its local heading, which is about 90 degrees
    # from the A3 asset's forward axis. Rotate translations by the same heading
    # correction used for root orientation; otherwise the task becomes a
    # sideways slide even though the robot visually faces forward.
    full_correction = Rotation.from_quat(stand_quat) * Rotation.from_quat(walk_quat[0]).inv()
    yaw_correction = full_correction.as_euler("zyx")[0]
    heading_correction = Rotation.from_euler("z", yaw_correction)
    root_delta_aligned = heading_correction.apply(walk_root - walk_root[0])
    root = stand_root + root_scale * root_delta_aligned
    quat = aligned_quaternions(stand_quat, walk_quat, pose_scale)

    # Add an explicit stationary-to-walking transition. This makes frame zero
    # exactly match the saved standing policy instead of presenting a pose jump.
    alpha = np.linspace(0.0, 1.0, n_transition, dtype=np.float32)
    ease = alpha * alpha * (3.0 - 2.0 * alpha)
    dof_t = stand_dof[None] + ease[:, None] * (dof[0] - stand_dof)[None]
    pose_t = stand_pose[None] + ease[:, None, None] * (pose[0] - stand_pose)[None]
    smpl_t = stand_smpl[None] + ease[:, None, None] * (smpl[0] - stand_smpl)[None]
    root_t = np.repeat(stand_root[None], n_transition, axis=0)
    quat_t = np.repeat(stand_quat[None], n_transition, axis=0)

    key = output_path.stem
    output = {
        key: {
            "root_trans_offset": np.concatenate([root_t, root]).astype(np.float32),
            "pose_aa": np.concatenate([pose_t, pose]).astype(np.float32),
            "dof": np.concatenate([dof_t, dof]).astype(np.float32),
            "root_rot": np.concatenate([quat_t, quat]).astype(np.float32),
            "smpl_joints": np.concatenate([smpl_t, smpl]).astype(np.float32),
            "fps": fps,
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, output_path)
    print(f"saved {output_path} ({len(output[key]['dof'])} frames @ {fps} fps)")
    print("first pose max error from stand:", np.abs(output[key]["dof"][0] - stand_dof).max())


if __name__ == "__main__":
    main()
