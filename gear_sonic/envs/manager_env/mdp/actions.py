from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass

# Joint ordering constants
G1_MUJOCO_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = None


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Add a bounded policy residual to the current motion-reference PD target."""

    cfg: "ReferenceResidualJointPositionActionCfg"

    def process_actions(self, actions):
        command = self._env.command_manager.get_term(self.cfg.command_name)
        reference_joint_pos = command.joint_pos[:, self._joint_ids]
        reference_action = (reference_joint_pos - self._offset) / self._scale
        # Keep the policy output before safety clipping.  The environment uses
        # the clipped residual, while reward terms may still penalize attempts
        # to drive the actuator beyond the configured residual envelope.
        self.raw_residual_actions = actions
        residual_actions = actions
        if self.cfg.residual_clip is not None:
            residual_actions = residual_actions.clamp(
                min=-self.cfg.residual_clip,
                max=self.cfg.residual_clip,
            )
        combined_action = reference_action + self.cfg.residual_scale * residual_actions

        # Keep both signals available for rollout collection and diagnostics.
        self.reference_actions = reference_action
        self.residual_actions = residual_actions
        super().process_actions(combined_action)


@configclass
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for reference-plus-residual joint-position control."""

    class_type: type = ReferenceResidualJointPositionAction
    command_name: str = "motion"
    residual_scale: float = 0.25
    residual_clip: float | None = None


class BoundedJointPositionAction(JointPositionAction):
    """Joint-position action with an explicit raw policy-action envelope.

    Isaac Lab's built-in ``clip`` is applied after scale and offset, while the
    A3 deployment contract clips the normalized policy output first.  Keeping
    that distinction is essential for reproducing an external controller and
    for retaining its pre-clip action in the next observation frame.
    """

    cfg: "BoundedJointPositionActionCfg"

    def __init__(self, cfg: "BoundedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._raw_clip = torch.tensor(
            [[-float("inf"), float("inf")]], device=self.device
        ).repeat(self.num_envs, self.action_dim, 1)
        if cfg.raw_clip:
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.raw_clip, self._joint_names
            )
            self._raw_clip[:, index_list] = torch.tensor(value_list, device=self.device)
        self.unclipped_raw_actions = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions):
        self.unclipped_raw_actions[:] = actions
        bounded = torch.clamp(
            actions,
            min=self._raw_clip[:, :, 0],
            max=self._raw_clip[:, :, 1],
        )
        super().process_actions(bounded)


@configclass
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for normalized-action clipping before affine scaling."""

    class_type: type = BoundedJointPositionAction
    raw_clip: dict[str, tuple[float, float]] | None = None
