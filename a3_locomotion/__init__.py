"""Isaac Lab velocity-locomotion task for the Zhiyuan A3."""

import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Flat-A3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "a3_locomotion.env_cfg:A3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "a3_locomotion.runner_cfg:A3FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-A3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "a3_locomotion.env_cfg:A3FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "a3_locomotion.runner_cfg:A3FlatPPORunnerCfg",
    },
)
