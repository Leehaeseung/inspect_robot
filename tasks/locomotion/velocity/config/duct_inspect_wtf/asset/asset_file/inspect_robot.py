import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import math
import random
##
# Configuration
##

INSPECT_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/IsaacLabExtensionTemplate/exts/ext_template/ext_template/tasks/locomotion/velocity/config/duct_inspect_wtf/asset/inspect_robot_flatten.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            solver_velocity_iteration_count=16,
            solver_position_iteration_count=4,
            max_depenetration_velocity=1.0,
        )
        ,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True, enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=2
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "rb_joint" :0.0,
            "rf_joint" :0.0,
            "lb_joint" :0.0,
            "lf_joint" :0.0,
            "lift_joint1" :0.0,
            "lift_joint2" :0.0,
            "lift_joint3" :0.0,
            "camera_joint1" :0.0,
            "camera_joint2" :0.0,
            "camera_joint3" :0.0,
            "liftt_assist_joint" :0.0,
            
        },
    ),
    actuators={
        "wheels" : ImplicitActuatorCfg(
            joint_names_expr=["rb_joint", "lf_joint", "rf_joint", "lb_joint"],
            effort_limit=30000.0,
            velocity_limit= 200.0,
            stiffness=0,
            damping=1e5,
        ),
        "joints" : ImplicitActuatorCfg(
        joint_names_expr=["lift_joint1","camera_joint2","camera_joint3"],
        effort_limit=0.00,
        velocity_limit=0.0,
        stiffness=1e20,
        damping=1e20,
        friction=1e20,
        )
    },
    soft_joint_pos_limit_factor=0.9,
)