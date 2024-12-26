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
            max_depenetration_velocity=5.0,
        )
        ,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=2
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "rb_wheel" :0.0,
            "rf_wheel" :0.0,
            "lb_wheel" :0.0,
            "lf_wheel" :0.0
        },
    ),
    actuators={
        "wheels" : ImplicitActuatorCfg(
            joint_names_expr=["rb_joint", "lf_joint", "rf_joint", "lb_joint"],
            effort_limit=3402823466385288598117041834845.0,
            velocity_limit= 10000000.0,
            stiffness=0,
            damping=1e5,
        )
        "lift_joint" : 
    },
    soft_joint_pos_limit_factor=1.0,
)