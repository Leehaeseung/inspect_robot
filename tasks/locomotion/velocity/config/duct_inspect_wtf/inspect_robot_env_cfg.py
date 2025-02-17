from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp
from omni.isaac.lab.sensors import TiledCameraCfg

from ext_template.tasks.locomotion.velocity.inspect_robot_env_cfg import InspectRobotbaseSceneCfg, InspectRobotDuctEnvCfg

from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg




@configclass
class InspectCameraEnvCfg(InspectRobotbaseSceneCfg):

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link3/Camera/Camera1",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0,0,0),rot=(0,1,0,0)),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=320,
        height=320,
    )
    
@configclass
class InspectRGBCameraEnvCfg(InspectRobotDuctEnvCfg):
    """Configuration for the cartpole environment with RGB camera."""

    scene: InspectRobotbaseSceneCfg = InspectCameraEnvCfg(num_envs=64, env_spacing=4)