"""Flat-ground velocity curriculum for A3.

This deliberately starts with slow forward motion only.  It is independent of
motion references and therefore cannot solve the task by copying joint angles.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

from gear_sonic.envs.manager_env.robots.a3 import A3_ACTION_SCALE, A3_CFG
from . import mdp as a3_mdp


@configclass
class A3FlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = A3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # A3 uses elbow_joint and has no finger joints.
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
        )
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_.*_joint"]
        )

        # Stage 1: every environment receives a small non-zero forward command.
        # This removes the stationary optimum while keeping impacts conservative.
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.32)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Preserve SONIC's per-joint action contract: one quarter of the static
        # effort limit divided by the SI stiffness. Changing this scale also
        # changes checkpoint semantics, independently of gain-unit handling.
        self.actions.joint_pos.scale = dict(A3_ACTION_SCALE)

        # A clock observable makes the desired left/right contact mode learnable.
        self.observations.policy.gait_phase = ObsTerm(
            func=a3_mdp.gait_phase, params={"period": 0.8}
        )
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.rewards.gait_contact_match = RewTerm(
            func=a3_mdp.gait_contact_match,
            weight=2.5,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet),
                "period": 0.8,
            },
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=a3_mdp.swing_foot_clearance,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=feet),
                "period": 0.8,
                "target_height": 0.06,
                "std": 0.035,
            },
        )
        self.rewards.root_height_l2 = RewTerm(
            func=a3_mdp.root_height_l2,
            weight=-5.0,
            params={
                "target_height": 1.03,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_slide.weight = -0.30
        self.rewards.flat_orientation_l2.weight = -2.0

        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class A3FlatEnvCfg_PLAY(A3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.rel_standing_envs = 0.0
