#!/usr/bin/env python3
"""Validate A3 reference-PD tracking and collect physics-aware teacher rollouts.

The motion-library ``dof`` field is stored in A3 MuJoCo joint-name order.  This
script maps it to the articulation order by name, executes the reference as an
Isaac Lab position target, and records the *actual* simulated state alongside
the reference and action target.  Samples after a fall are deliberately not
written.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--motion-key", type=str)
parser.add_argument("--output", type=Path)
parser.add_argument("--future-frames", type=int, default=10)
parser.add_argument("--control-hz", type=float, default=50.0)
parser.add_argument("--min-root-height", type=float, default=0.72)
parser.add_argument("--max-root-tilt", type=float, default=0.65)
parser.add_argument(
    "--fast-exit",
    action="store_true",
    help="Exit immediately after flushing output; avoids slow Isaac plugin teardown in batch jobs.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import joblib
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from gear_sonic.envs.manager_env.robots.a3 import A3_CFG, A3_MUJOCO_DOF_NAMES


class A3TeacherSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    robot = A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def load_motion(path: Path, key: str | None) -> tuple[str, dict]:
    motions = joblib.load(path)
    if key is None:
        if len(motions) != 1:
            raise ValueError(f"{path} contains {len(motions)} motions; pass --motion-key")
        key = next(iter(motions))
    if key not in motions:
        raise KeyError(f"{key!r} not found; available keys: {sorted(motions)}")
    motion = motions[key]
    dof = np.asarray(motion["dof"], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != len(A3_MUJOCO_DOF_NAMES):
        raise ValueError(f"expected dof shape [T,29], got {dof.shape}")
    return key, motion


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.empty_like(values)
    velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    velocity[0] = (values[1] - values[0]) / dt
    velocity[-1] = (values[-1] - values[-2]) / dt
    return velocity


def future_window(values: np.ndarray, frame: int, count: int) -> np.ndarray:
    indices = np.minimum(np.arange(frame, frame + count), len(values) - 1)
    return values[indices]


def main() -> None:
    motion_key, motion = load_motion(args.motion, args.motion_key)
    ref_q_mujoco = np.asarray(motion["dof"], dtype=np.float32)
    ref_root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    ref_root_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    motion_hz = float(motion["fps"])
    motion_dt = 1.0 / motion_hz
    ref_dq_mujoco = finite_difference(ref_q_mujoco, motion_dt)

    sim_dt = 0.005
    sim = SimulationContext(SimulationCfg(dt=sim_dt, device=args.device))
    scene = InteractiveScene(A3TeacherSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    if set(robot.joint_names) != set(A3_MUJOCO_DOF_NAMES):
        missing = sorted(set(A3_MUJOCO_DOF_NAMES) - set(robot.joint_names))
        extra = sorted(set(robot.joint_names) - set(A3_MUJOCO_DOF_NAMES))
        raise RuntimeError(f"A3 joint schema mismatch: missing={missing}, extra={extra}")
    mujoco_to_articulation = np.asarray(
        [A3_MUJOCO_DOF_NAMES.index(name) for name in robot.joint_names], dtype=np.int64
    )
    ref_q = ref_q_mujoco[:, mujoco_to_articulation]
    ref_dq = ref_dq_mujoco[:, mujoco_to_articulation]

    device = robot.device
    default_q = robot.data.default_joint_pos[0].detach().cpu().numpy()
    initial_root_pos = ref_root_pos[0].copy()
    # Keep the motion's x/y origin but use at least the configured safe spawn height.
    initial_root_pos[2] = max(initial_root_pos[2], float(A3_CFG.init_state.pos[2]))
    initial_root_wxyz = ref_root_xyzw[0, [3, 0, 1, 2]]
    root_pose = torch.tensor(
        np.concatenate([initial_root_pos, initial_root_wxyz]),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    robot.write_joint_state_to_sim(
        torch.tensor(ref_q[0], device=device).unsqueeze(0),
        torch.tensor(ref_dq[0], device=device).unsqueeze(0),
    )
    robot.set_joint_position_target(torch.tensor(ref_q[0], device=device).unsqueeze(0))
    scene.write_data_to_sim()
    sim.step(render=not args.headless)
    scene.update(sim_dt)

    records: dict[str, list[np.ndarray | float | int]] = {
        "joint_pos": [],
        "joint_vel": [],
        "root_pos": [],
        "root_quat_wxyz": [],
        "projected_gravity": [],
        "reference_joint_pos": [],
        "reference_joint_vel": [],
        "reference_joint_pos_future": [],
        "reference_joint_vel_future": [],
        "action": [],
        "frame": [],
    }
    elapsed = 0.0
    next_record_time = 0.0
    fall_reason = "completed"
    last_frame = 0
    total_time = (len(ref_q) - 1) * motion_dt
    while elapsed <= total_time:
        ref_float = elapsed * motion_hz
        frame0 = min(int(np.floor(ref_float)), len(ref_q) - 1)
        frame1 = min(frame0 + 1, len(ref_q) - 1)
        alpha = ref_float - frame0
        target_q_np = (1.0 - alpha) * ref_q[frame0] + alpha * ref_q[frame1]
        target_q = torch.tensor(target_q_np, device=device).unsqueeze(0)
        robot.set_joint_position_target(target_q)
        scene.write_data_to_sim()
        sim.step(render=not args.headless)
        scene.update(sim_dt)
        elapsed += sim_dt
        last_frame = frame0

        height = float(robot.data.root_pos_w[0, 2])
        tilt = float(torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2]))
        if height < args.min_root_height:
            fall_reason = f"root_height<{args.min_root_height}"
            break
        if tilt > args.max_root_tilt:
            fall_reason = f"root_tilt>{args.max_root_tilt}"
            break

        if elapsed + 1.0e-9 < next_record_time:
            continue
        next_record_time += 1.0 / args.control_hz
        q_actual = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        records["joint_pos"].append(q_actual)
        records["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        records["root_pos"].append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
        records["root_quat_wxyz"].append(
            robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        )
        records["projected_gravity"].append(
            robot.data.projected_gravity_b[0].detach().cpu().numpy().copy()
        )
        records["reference_joint_pos"].append(ref_q[frame0].copy())
        records["reference_joint_vel"].append(ref_dq[frame0].copy())
        records["reference_joint_pos_future"].append(
            future_window(ref_q, frame0, args.future_frames)
        )
        records["reference_joint_vel_future"].append(
            future_window(ref_dq, frame0, args.future_frames)
        )
        # Exact raw JointPositionAction target when use_default_offset=true.
        records["action"].append(target_q_np.copy() - default_q)
        records["frame"].append(frame0)

    arrays = {key: np.asarray(value) for key, value in records.items()}
    tracked_frames = int(last_frame + 1)
    position_rmse = (
        float(np.sqrt(np.mean((arrays["joint_pos"] - arrays["reference_joint_pos"]) ** 2)))
        if len(arrays["joint_pos"])
        else float("nan")
    )
    print(
        "A3_TEACHER "
        f"motion={motion_key} source_frames={len(ref_q)} tracked_frames={tracked_frames} "
        f"samples={len(arrays['frame'])} duration={elapsed:.3f}s "
        f"joint_rmse={position_rmse:.6f} result={fall_reason}",
        flush=True,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            **arrays,
            motion_key=np.asarray(motion_key),
            source_motion=np.asarray(str(args.motion)),
            joint_names=np.asarray(robot.joint_names),
            motion_fps=np.asarray(motion_hz),
            control_hz=np.asarray(args.control_hz),
            completed=np.asarray(fall_reason == "completed"),
            termination_reason=np.asarray(fall_reason),
        )
        print(f"saved={args.output}", flush=True)
    if args.fast_exit:
        os._exit(0)
    app.close()


if __name__ == "__main__":
    main()
