from __future__ import annotations

import torch
from typing import TYPE_CHECKING


from omni.isaac.lab.assets import RigidObject,Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def track_pos_x_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str,    
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command=env.command_manager.get_command(command_name)
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    pos_x_error = torch.square(des_pos_w[:,0] - robot.data.root_pos_w[:, 0])
    return torch.exp(-pos_x_error/std**2)


def time_out_penalty(
    env: ManagerBasedRLEnv 
    ):
    
    return env.episode_length_buf >= env.max_episode_length-0.1

def reach_goal_reward(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    command_name: str,    
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command=env.command_manager.get_command(command_name)
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    pos_x_error = torch.square(des_pos_w[:,0] - robot.data.root_pos_w[:, 0])
    return (pos_x_error<threshold)



def track_pos_y_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str,    
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    duct_cfg: SceneEntityCfg = SceneEntityCfg("duct")
    
) -> torch.Tensor:
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    duct : RigidObject=env.scene[duct_cfg.name]
    command=env.command_manager.get_command(command_name)
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # des_pos_b=command[:,:3]
    # des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # pos_x_error = torch.norm(des_pos_w[:,0] - object.data.root_pos_w[:, 0], dim=1)
    pos_y_error=torch.square(des_pos_w[:,1]-robot.data.root_pos_w[:,1])
    return torch.exp(-pos_y_error/std**2)


def position_command_error_tanh_x(env: ManagerBasedRLEnv,
                                  std: float,
                                  command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                                  ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command = env.command_manager.get_command(command_name)
    # des_pos_b = command[:, 0]
    # distance = torch.norm(des_pos_b-robot.data.root_pos_w[:,0], dim=1)
    # des_pos_b = command[:, 0]  
    cur_pos_x = robot.data.root_pos_w[:, 0]  
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = (des_pos_w[:,0] - cur_pos_x).abs()  
    return 1 - torch.tanh(distance / std)

def position_command_error_linear_x(env: ManagerBasedRLEnv,
                                  std: float,
                                  command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                                  ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command = env.command_manager.get_command(command_name)
    # des_pos_b = command[:, 0]
    # distance = torch.norm(des_pos_b-robot.data.root_pos_w[:,0], dim=1)
    # des_pos_b = command[:, 0]  
    cur_pos_x = robot.data.root_pos_w[:, 0]  
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = (des_pos_w[:,0] - cur_pos_x).abs()  
    return 1 - (distance / std)




def orientation_tracking(
    env: ManagerBasedRLEnv, 
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    return 1- torch.tanh(torch.acos(-asset.data.projected_gravity_b[:, 2]).abs()/std)


def joint_torques_l2_only_inspect_robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, [0,1,4,5]]), dim=1)


def root_height_below_minimum_penalty(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < minimum_height)

def velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.where(torch.abs(asset.data.joint_vel[:, [0,1,4,5]]) > 6.28, torch.square(asset.data.joint_vel[:, [0,1,4,5]]), 0.0).max(dim=1)[0]

def vel_action_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.where(torch.abs(env.action_manager.action[:, [0,1]]) > 6.28, torch.square(env.action_manager.action[:, [0,1]]), 0.0).max(dim=1)[0]





def position_command_error_tanh_y(env: ManagerBasedRLEnv,
                                  std: float,
                                  command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                                  ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command = env.command_manager.get_command(command_name)
    # des_pos_b = command[:, 1]
    cur_pos_y = robot.data.root_pos_w[:, 1]
    des_pos_b=command[:,:3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = (des_pos_w[:,1] - cur_pos_y).abs()  

    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv,
                            command_name: str,
                            std: float,                            
                            robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        
) -> torch.Tensor:
    """Penalize tracking orientation error."""
    robot:RigidObject=env.scene[robot_cfg.name]    # compute the error
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    cur_head_w= robot.data.heading_w[:]
    heading_error=(heading_b-cur_head_w).abs()
    
    return 1 - torch.tanh(heading_error / std )


def lin_vel_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.where(asset.data.root_lin_vel_w[:, 0] > 0.5, -50.0, 0.0)
    return asset.data.root_lin_vel_w[:, 0]*20+penalty