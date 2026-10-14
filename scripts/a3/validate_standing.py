#!/usr/bin/env python3
"""Run a zero-action PD standing check for the SONIC A3 articulation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg

from gear_sonic.envs.manager_env.robots.a3 import A3_CFG


class StandingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    robot = A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    dt = 0.005
    sim = SimulationContext(SimulationCfg(dt=dt, device=args.device))
    scene = InteractiveScene(StandingSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    target = robot.data.default_joint_pos.clone()
    heights = []
    tilts = []
    max_torque_ratio = 0.0
    for _ in range(round(args.seconds / dt)):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        heights.append(float(robot.data.root_pos_w[0, 2]))
        # Projected gravity x/y magnitude is zero for an upright pelvis.
        tilts.append(float(torch.linalg.vector_norm(robot.data.projected_gravity_b[0, :2])))
        effort_limits = robot.data.joint_effort_limits[0].clamp_min(1.0e-6)
        torque_ratio = robot.data.applied_torque[0].abs() / effort_limits
        max_torque_ratio = max(max_torque_ratio, float(torque_ratio.max()))
    stiffness = robot.actuators["legs"].stiffness[0]
    damping = robot.actuators["legs"].damping[0]
    print(
        "A3_STAND "
        f"seconds={args.seconds:.2f} final_height={heights[-1]:.4f} "
        f"min_height={min(heights):.4f} final_tilt={tilts[-1]:.4f} "
        f"max_tilt={max(tilts):.4f} max_torque_ratio={max_torque_ratio:.4f} "
        f"leg_kp_min={float(stiffness.min()):.3f} "
        f"leg_kp_max={float(stiffness.max()):.3f} "
        f"leg_kd_min={float(damping.min()):.3f} "
        f"leg_kd_max={float(damping.max()):.3f}"
    )
    app.close()


if __name__ == "__main__":
    main()
