from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp

from ext_template.tasks.locomotion.velocity.inspect_robot_env_cfg import InspectRobotbaseSceneCfg, InspectRobotDuctEnvCfg

from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg




@configclass
class InspectCameraEnvCfg(InspectRobotbaseSceneCfg):
    camera=CameraCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera1",
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera/Camera1",
        offset=CameraCfg.OffsetCfg(pos=(0.0,0,0),rot=(0,1,0,0)),
        width=80,
        height=80,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10000000.0)
        )
    )    

    
@configclass
class InspectRGBCameraEnvCfg(InspectRobotDuctEnvCfg):
    """Configuration for the cartpole environment with RGB camera."""

    scene: InspectRobotbaseSceneCfg = InspectCameraEnvCfg(num_envs=64, env_spacing=3)