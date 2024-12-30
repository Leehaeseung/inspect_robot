from __future__ import annotations

import math
from dataclasses import MISSING
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from omni.isaac.lab.assets import RigidObjectCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from ext_template.tasks.locomotion.velocity.config.duct_inspect_wtf.asset.asset_file.inspect_robot import INSPECT_ROBOT_CFG
import ext_template.tasks.locomotion.velocity.mdp as mdp



@configclass
class InspectRobotbaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
        # plane = AssetBaseCfg(
        #     prim_path="/World/GroundPlane",
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.115]),
        #     spawn=GroundPlaneCfg(),
        # )

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = INSPECT_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # duct
    duct = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/duct",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.2,0,-0.072], rot=[-0.7071,0,0,0.7071]),
        spawn=UsdFileCfg(usd_path="/home/ubuntu/IsaacLabExtensionTemplate/exts/ext_template/ext_template/tasks/locomotion/velocity/config/duct_inspect_wtf/asset/damper_winded.usd",
                         scale=(0.01,0.01,0.01),
                         activate_contact_sensors=False,
                         rigid_props=RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=0,
                            max_angular_velocity=0,
                            max_linear_velocity=0,
                            max_depenetration_velocity=1.0,
                            disable_gravity=True),

                         )
        )
    

    #충돌센서(ductside에만 붙일거임)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                      filter_prim_paths_expr=["{ENV_REGEX_NS}/duct"],
                                      track_air_time=True,
                                      debug_vis=True)

    # # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )

    

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # base_position=mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="base_link",
    #     resampling_time_range=(10.0,10.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(12.5,12.5), pos_y=(0.0,0.0), pos_z=(-0.1,-0.1),roll=(0,0),pitch=(0,0),yaw=(0,0)
    #     )
    # )


    base_position=mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(15.0,15.0),
        debug_vis=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(2.0,2.0), pos_y=(0.0,0.0), heading=(0,0)#rad
        )
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    wheel_velocity=mdp.JointVelocityActionCfg(asset_name="robot",joint_names=["rb_joint", "lb_joint"],scale=1.0,use_default_offset=False)
    




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"})
        actions = ObsTerm(func=mdp.last_action)
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"yaw":(4.612,4.812)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot")
        }
    )


# 보상함수 설계 부분

@configclass
class RewardsCfg:
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp,# track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    
    reach_goal= RewTerm(
        func=mdp.reach_goal_reward,
        weight=1.0,
        params={"command_name":"base_position","threshold": 0.1},
    )
    time_out=RewTerm(
        func=mdp.time_out_penalty,
        weight=-1.5,
        )
    
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=2.0,
    #     params={"command_name": "base_position","std":0.05*math.pi},
    # )#완
    
    # position_tracking_fine_grained = RewTerm(
    # func=mdp.position_command_error_tanh,
    # weight=0.5,
    # params={"std": 0.2, "command_name": "base_position"},
    # )
    # 초기 넓은 탐색 (큰 std, 높은 weight) 이게 x,y 커맨드 출력이 뭔지 모르겠네
    position_tracking_fine_grained_3_0 = RewTerm(
        func=mdp.position_command_error_tanh_x,
        weight=3.0,  # 높은 가중치로 초기 탐색 지원
        params={"std": 3.0, "command_name": "base_position"},
    )

    position_tracking_fine_grained_1_5 = RewTerm(
        func=mdp.position_command_error_tanh_x,
        weight=1.8,  # 초기 탐색, 조금 더 낮은 가중치
        params={"std": 1.5, "command_name": "base_position"},
    )#완

    # 중간 범위 탐색 (중간 std, 중간 weight)
    position_tracking_fine_grained_0_8 = RewTerm(
        func=mdp.position_command_error_tanh_x,
        weight=1.0,  # 중간 가중치로 학습 안정화
        params={"std": 1.0, "command_name": "base_position"},
    )#완
    
    #y축 중간으로
    # position_tracking_fine_grained_0_8_y = RewTerm(
    #     func=mdp.position_command_error_tanh_y,
    #     weight=1.0,  # 중간 가중치로 학습 안정화
    #     params={"std": 0.03, "command_name": "base_position"},
    # )#완

    # 목표 근처 정밀 제어 (작은 std, 높은 weight)
    position_tracking_fine_grained_0_4 = RewTerm(
        func=mdp.position_command_error_tanh_x,
        weight=0.9,  # 정밀 제어에 적당한 가중치
        params={"std": 0.8, "command_name": "base_position"},
    )

    position_tracking_fine_grained_0_2 = RewTerm(
        func=mdp.position_command_error_tanh_x,
        weight=0.7,  # 높은 가중치로 정밀 제어 강화
        params={"std": 0.5, "command_name": "base_position"},
    )

    





@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    robot_dropping = DoneTerm(
        func=mdp.root_X_below_minimum, params={"minimum_height": -2.0,"asset_cfg": SceneEntityCfg("robot")}
    )
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    reach_goal=DoneTerm(func=mdp.reach_goal_termination, params={"command_name":"base_position","threshold": 0.1})
        
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},

    )
    # 도착했을때 상점주면서 집 보내기
    
    
    
# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     pos_x_reward = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "orientation_tracking", "weight": -1e-1, "num_steps": 1000}
#     )





@configclass
class InspectRobotDuctEnvCfg(ManagerBasedRLEnvCfg):    
    scene: InspectRobotbaseSceneCfg = InspectRobotbaseSceneCfg(num_envs=64, env_spacing=4)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
