from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

def x_bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle

def reach_goal_termination(env: ManagerBasedRLEnv, 
    command_name: str,    
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):

    
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command=env.command_manager.get_command(command_name)
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    pos_x_error = torch.square(des_pos_w[:,0] - robot.data.root_pos_w[:, 0])
    return (pos_x_error<threshold)
