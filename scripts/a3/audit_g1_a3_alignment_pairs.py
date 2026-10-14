#!/usr/bin/env python3
"""Audit paired G1/A3 motion libraries before token alignment.

The audit matches files by relative path and motions by key.  G1 DOFs are
reordered into A3 MuJoCo joint-name order before comparison, so a coincidental
29-DOF shape match cannot pass as semantic compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mujoco
import numpy as np


REQUIRED_FIELDS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "fps")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def joint_names(model: mujoco.MjModel) -> list[str]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    if any(name is None for name in names):
        raise RuntimeError("all non-root MuJoCo joints must be named")
    return names


def motion_files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*.pkl"))
        if path.name != "metadata.pkl"
    }


def load_motion_file(path: Path) -> dict[str, dict]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"{path}: expected a non-empty motion dictionary")
    invalid = [key for key, entry in payload.items() if not isinstance(entry, dict)]
    if invalid:
        raise RuntimeError(f"{path}: non-motion entries: {invalid[:5]}")
    return payload


def finite_entry(entry: dict) -> bool:
    return all(
        np.isfinite(np.asarray(value)).all()
        for key, value in entry.items()
        if key != "fps" and hasattr(value, "shape")
    )


def quaternion_angle_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(1.0e-12)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(1.0e-12)
    dots = np.abs(np.sum(first * second, axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(dots)


def canonical_source_name(motion_key: str) -> str:
    return motion_key[:-2] if motion_key.endswith("_M") else motion_key


def audit_pair(
    relative_path: str,
    motion_key: str,
    g1_path: Path,
    a3_path: Path,
    g1_entry: dict,
    a3_entry: dict,
    g1_to_a3_dof: list[int],
    target_fps: int,
    num_future_frames: int,
    future_dt: float,
    tolerance: float,
) -> dict:
    missing_g1 = sorted(set(REQUIRED_FIELDS) - set(g1_entry))
    missing_a3 = sorted(set(REQUIRED_FIELDS) - set(a3_entry))
    if missing_g1 or missing_a3:
        return {
            "relative_path": relative_path,
            "motion_key": motion_key,
            "ready": False,
            "error": {"g1_missing": missing_g1, "a3_missing": missing_a3},
        }

    g1_dof = np.asarray(g1_entry["dof"], dtype=np.float64)
    a3_dof = np.asarray(a3_entry["dof"], dtype=np.float64)
    g1_root_pos = np.asarray(g1_entry["root_trans_offset"], dtype=np.float64)
    a3_root_pos = np.asarray(a3_entry["root_trans_offset"], dtype=np.float64)
    g1_root_rot = np.asarray(g1_entry["root_rot"], dtype=np.float64)
    a3_root_rot = np.asarray(a3_entry["root_rot"], dtype=np.float64)
    g1_pose = np.asarray(g1_entry["pose_aa"], dtype=np.float64)
    a3_pose = np.asarray(a3_entry["pose_aa"], dtype=np.float64)

    frame_counts = {
        "g1_dof": len(g1_dof),
        "a3_dof": len(a3_dof),
        "g1_root_pos": len(g1_root_pos),
        "a3_root_pos": len(a3_root_pos),
        "g1_root_rot": len(g1_root_rot),
        "a3_root_rot": len(a3_root_rot),
        "g1_pose": len(g1_pose),
        "a3_pose": len(a3_pose),
    }
    same_frames = len(set(frame_counts.values())) == 1
    same_fps = float(g1_entry["fps"]) == float(a3_entry["fps"])
    shape_ok = (
        g1_dof.ndim == 2
        and a3_dof.ndim == 2
        and g1_dof.shape[1] == len(g1_to_a3_dof)
        and a3_dof.shape[1] == len(g1_to_a3_dof)
        and g1_root_pos.shape[1:] == (3,)
        and a3_root_pos.shape[1:] == (3,)
        and g1_root_rot.shape[1:] == (4,)
        and a3_root_rot.shape[1:] == (4,)
    )
    finite = finite_entry(g1_entry) and finite_entry(a3_entry)

    if same_frames and shape_ok:
        semantic_g1_dof = g1_dof[:, g1_to_a3_dof]
        dof_error = np.abs(a3_dof - semantic_g1_dof)
        root_xy_error = np.abs(a3_root_pos[:, :2] - g1_root_pos[:, :2])
        root_z_delta = a3_root_pos[:, 2] - g1_root_pos[:, 2]
        root_angle_error = quaternion_angle_error(a3_root_rot, g1_root_rot)
        root_pose_error = np.abs(a3_pose[:, 0] - g1_pose[:, 0])
        metrics = {
            "semantic_dof_mae_rad": float(dof_error.mean()),
            "semantic_dof_max_abs_rad": float(dof_error.max()),
            "semantic_dof_changed_fraction": float(np.mean(dof_error > tolerance)),
            "root_xy_max_abs_m": float(root_xy_error.max()),
            "root_z_delta_mean_m": float(root_z_delta.mean()),
            "root_z_delta_std_m": float(root_z_delta.std()),
            "root_quaternion_max_angle_rad": float(root_angle_error.max()),
            "root_pose_rotvec_max_abs_rad": float(root_pose_error.max()),
        }
    else:
        metrics = None

    source_frames = next(iter(frame_counts.values())) if same_frames else 0
    source_fps = float(g1_entry["fps"]) if same_fps else 0.0
    if source_frames > 1 and source_fps > 0:
        if np.isclose(source_fps, target_fps, atol=1.0e-9, rtol=0.0):
            target_frames = source_frames
        else:
            duration = (source_frames - 1) / source_fps
            target_frames = int(len(np.arange(0.0, duration, 1.0 / target_fps)))
    else:
        target_frames = 0
    future_step = int(round(future_dt * target_fps))
    valid_windows = max(0, target_frames - (num_future_frames - 1) * future_step)

    ready = bool(
        same_frames
        and same_fps
        and shape_ok
        and finite
        and metrics is not None
        and metrics["root_xy_max_abs_m"] <= tolerance
        and metrics["root_quaternion_max_angle_rad"] <= tolerance
        and metrics["root_pose_rotvec_max_abs_rad"] <= tolerance
        and valid_windows > 0
    )
    return {
        "relative_path": relative_path,
        "motion_key": motion_key,
        "canonical_source_motion": canonical_source_name(motion_key),
        "mirrored": motion_key.endswith("_M"),
        "ready": ready,
        "g1_file": str(g1_path.resolve()),
        "a3_file": str(a3_path.resolve()),
        "g1_sha256": file_sha256(g1_path),
        "a3_sha256": file_sha256(a3_path),
        "frame_counts": frame_counts,
        "fps": {"g1": float(g1_entry["fps"]), "a3": float(a3_entry["fps"])},
        "shape_ok": shape_ok,
        "finite": finite,
        "target_frames_at_alignment_fps": target_frames,
        "valid_alignment_windows": valid_windows,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--a3-root", type=Path, required=True)
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
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--future-dt", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    if args.target_fps <= 0 or args.num_future_frames <= 0 or args.future_dt <= 0:
        parser.error("target-fps, num-future-frames and future-dt must be positive")
    future_steps = args.future_dt * args.target_fps
    if not np.isclose(future_steps, round(future_steps), atol=1.0e-9, rtol=0.0):
        parser.error("future-dt * target-fps must be an integer")

    g1_model = mujoco.MjModel.from_xml_path(str(args.g1_mjcf.resolve()))
    a3_model = mujoco.MjModel.from_xml_path(str(args.a3_mjcf.resolve()))
    g1_names = joint_names(g1_model)
    a3_names = joint_names(a3_model)
    if set(g1_names) != set(a3_names):
        raise RuntimeError(
            "G1/A3 joint-name sets differ: "
            f"missing_in_g1={sorted(set(a3_names) - set(g1_names))} "
            f"missing_in_a3={sorted(set(g1_names) - set(a3_names))}"
        )
    g1_to_a3_dof = [g1_names.index(name) for name in a3_names]

    g1_files = motion_files(args.g1_root)
    a3_files = motion_files(args.a3_root)
    common_files = sorted(set(g1_files) & set(a3_files))
    pairs = []
    file_errors = []
    for relative_path in common_files:
        try:
            g1_payload = load_motion_file(g1_files[relative_path])
            a3_payload = load_motion_file(a3_files[relative_path])
            common_keys = sorted(set(g1_payload) & set(a3_payload))
            for motion_key in common_keys:
                pairs.append(
                    audit_pair(
                        relative_path,
                        motion_key,
                        g1_files[relative_path],
                        a3_files[relative_path],
                        g1_payload[motion_key],
                        a3_payload[motion_key],
                        g1_to_a3_dof,
                        args.target_fps,
                        args.num_future_frames,
                        args.future_dt,
                        args.tolerance,
                    )
                )
            if set(g1_payload) != set(a3_payload):
                file_errors.append(
                    {
                        "relative_path": relative_path,
                        "g1_only_motion_keys": sorted(set(g1_payload) - set(a3_payload)),
                        "a3_only_motion_keys": sorted(set(a3_payload) - set(g1_payload)),
                    }
                )
        except Exception as exc:  # report all bad files in one audit
            file_errors.append({"relative_path": relative_path, "error": str(exc)})

    ready_pairs = [pair for pair in pairs if pair["ready"]]
    canonical_sources = sorted(
        {pair["canonical_source_motion"] for pair in ready_pairs}
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "contract": {
            "g1_root": str(args.g1_root.resolve()),
            "a3_root": str(args.a3_root.resolve()),
            "g1_mjcf": str(args.g1_mjcf.resolve()),
            "a3_mjcf": str(args.a3_mjcf.resolve()),
            "g1_mjcf_sha256": file_sha256(args.g1_mjcf),
            "a3_mjcf_sha256": file_sha256(args.a3_mjcf),
            "g1_mujoco_joint_names": g1_names,
            "a3_mujoco_joint_names": a3_names,
            "g1_to_a3_dof_indices": g1_to_a3_dof,
            "target_fps": args.target_fps,
            "num_future_frames": args.num_future_frames,
            "future_dt_seconds": args.future_dt,
            "future_step_at_target_fps": int(round(future_steps)),
            "token_shape": [2, 32],
        },
        "summary": {
            "g1_files": len(g1_files),
            "a3_files": len(a3_files),
            "common_relative_files": len(common_files),
            "g1_only_files": sorted(set(g1_files) - set(a3_files)),
            "a3_only_files": sorted(set(a3_files) - set(g1_files)),
            "motion_pairs": len(pairs),
            "ready_motion_pairs": len(ready_pairs),
            "independent_source_motions": len(canonical_sources),
            "canonical_source_motions": canonical_sources,
            "valid_alignment_windows": int(
                sum(pair["valid_alignment_windows"] for pair in ready_pairs)
            ),
            "pilot_contract_ready": bool(ready_pairs and not file_errors),
            "general_alignment_training_ready": len(canonical_sources) >= 10,
            "general_alignment_blocker": (
                None
                if len(canonical_sources) >= 10
                else "fewer than 10 independent paired source motions; mirrored clips do not count independently"
            ),
        },
        "file_errors": file_errors,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "G1_A3_ALIGNMENT_AUDIT "
        f"pairs={len(pairs)} ready={len(ready_pairs)} "
        f"independent={len(canonical_sources)} "
        f"windows={manifest['summary']['valid_alignment_windows']} "
        f"pilot_ready={manifest['summary']['pilot_contract_ready']} "
        f"general_ready={manifest['summary']['general_alignment_training_ready']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
