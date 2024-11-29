import sys

import numpy as np
from isaacsim import SimulationApp

BOT_STAGE_PATH = "/bot"
BOT_USD_PATH = "inspect_robot/complete_robot.usd"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "inspect_robot/damper.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)


import carb
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, prims, rotations, stage, viewports
from omni.isaac.nucleus import get_assets_root_path
from pxr import Gf

extensions.enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0]))
# viewports.create_viewport_for_camera(viewport_name="what",camera_prim_path="/bot/base_link/Camera")
# Loading the damper environment

prims.create_prim(
    BOT_STAGE_PATH,
    "Xform",
    position=np.array([0, 0, 0]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 0), 90)),
    usd_path=BOT_USD_PATH,
)

prims.create_prim(
    BACKGROUND_STAGE_PATH,
    "Xform",
    position=np.array([-0.16, -0.02,-0.13291]),
    orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(90, 0, 0), 90)),
    usd_path=BACKGROUND_USD_PATH,
    scale=[0.01,0.01,0.01]
)

simulation_app.update()


# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

simulation_context.play()

while simulation_app.is_running():

    # Run with a fixed step size
    simulation_context.step(render=True)

    # Tick the Publish/Subscribe JointState and Publish Clock nodes each frame
    # og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)

simulation_context.stop()
simulation_app.close()
