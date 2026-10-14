"""A3-specific dense gait signals for early locomotion learning."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _phase(env, period: float) -> torch.Tensor:
    time = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(time / period, 1.0)


def gait_phase(env, period: float = 0.8) -> torch.Tensor:
    """Observable periodic clock used to coordinate left/right support."""
    angle = 2.0 * math.pi * _phase(env, period)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def gait_contact_match(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
) -> torch.Tensor:
    """Reward the commanded alternating single-support contact pattern."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    left_stance = _phase(env, period) < 0.5
    desired = torch.stack((left_stance, ~left_stance), dim=-1)
    # Require the complete two-foot mode to match. Averaging per foot would
    # incorrectly give double support a constant 0.5 reward.
    match = torch.all(contact == desired, dim=-1).float()
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1) > 0.1
    return match * moving


def swing_foot_clearance(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    period: float = 0.8,
    target_height: float = 0.06,
    std: float = 0.035,
) -> torch.Tensor:
    """Track swing-foot height relative to the current stance ankle."""
    robot = env.scene[asset_cfg.name]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    left_stance = _phase(env, period) < 0.5
    stance_z = torch.where(left_stance, foot_z[:, 0], foot_z[:, 1])
    swing_z = torch.where(left_stance, foot_z[:, 1], foot_z[:, 0])
    error = swing_z - stance_z - target_height
    reward = torch.exp(-error.square() / (std * std))
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1) > 0.1
    return reward * moving


def root_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Keep the floating base near the stable A3 standing height."""
    robot = env.scene[asset_cfg.name]
    return (robot.data.root_pos_w[:, 2] - target_height).square()
