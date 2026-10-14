"""RSL-RL PPO settings for the A3 flat locomotion task."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class A3FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "a3_flat_locomotion"
        self.max_iterations = 1500
        self.save_interval = 100
        self.policy.init_noise_std = 0.35
