import sys

import numpy as np
from isaacsim import SimulationApp

BOT_STAGE_PATH = "/bot"
BOT_USD_PATH = "inspect_robot/complete_robot.usd"
BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "inspect_robot/damper.usd"
CAMERA_STAGE_PATH="/bot/base_link/Camera"
ROS_CAMERA_GRAPH_PATH = "/ROS_Camera"



CONFIG = {"renderer": "RayTracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)


import carb
import omni
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, prims, rotations, stage, viewports
from omni.isaac.nucleus import get_assets_root_path
from pxr import Gf, Usd, UsdGeom
from omni.kit.viewport.utility import get_active_viewport


extensions.enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)


viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0]))


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

keys = og.Controller.Keys
(ros_camera_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
            ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
            ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraInfoHelper"),
            ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("cameraHelperRgb.inputs:frameId", "sim_camera"),
            ("cameraHelperRgb.inputs:topicName", "rgb"),
            ("cameraHelperRgb.inputs:type", "rgb"),
            ("cameraHelperInfo.inputs:frameId", "sim_camera"),
            ("cameraHelperInfo.inputs:topicName", "camera_info"),
            ("cameraHelperDepth.inputs:frameId", "sim_camera"),
            ("cameraHelperDepth.inputs:topicName", "depth"),
            ("cameraHelperDepth.inputs:type", "depth"),
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_STAGE_PATH)]),
        ],
    },
)

og.Controller.evaluate_sync(ros_camera_graph)

simulation_app.update()

SD_GRAPH_PATH = "/Render/PostProcess/SDGPipeline"

viewport_api = get_active_viewport()

if viewport_api is not None:
    import omni.syntheticdata._syntheticdata as sd

    curr_stage = omni.usd.get_context().get_stage()

    
    with Usd.EditContext(curr_stage, curr_stage.GetSessionLayer()):

    
        rv_rgb = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)

    
        rgb_camera_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_rgb + "IsaacSimulationGate", viewport_api.get_render_product_path()
        )
        rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        depth_camera_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv_depth + "IsaacSimulationGate", viewport_api.get_render_product_path()
        )

    
        camera_info_gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            "PostProcessDispatch" + "IsaacSimulationGate", viewport_api.get_render_product_path()
        )



simulation_context.initialize_physics()

simulation_context.play()

while simulation_app.is_running():


    simulation_context.step(render=True)


simulation_context.stop()
simulation_app.close()
