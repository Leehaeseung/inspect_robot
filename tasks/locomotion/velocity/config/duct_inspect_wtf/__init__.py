import gymnasium as gym

from . import agents,inspect_robot_env_cfg



gym.register(
    id="haeseung_inspect_robot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": inspect_robot_env_cfg.InspectRGBCameraEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspectRGBPPORunnerCfg",
    },
)