import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()


simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils import extensions, stage

my_world = World(stage_units_in_meters=1.0)

jetbot_asset_path = r"C:\Users\xslee\AppData\Local\ov\pkg\isaac-sim-4.1.0\inspect_robot\complete_robot.usd"
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/bot",
        name="bot",
        wheel_dof_indices=[4],
        wheel_dof_names=["rb_wheel", "lb_wheel", "rf_wheel", "lf_wheel"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0, 0.0, 1.0]),
    )
)
damper_asset_path=r"C:\Users\xslee\AppData\Local\ov\pkg\isaac-sim-4.1.0\inspect_robot\damper.usd"

damper=my_world.scene.add(
    
)
# my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
# my_world.reset()
while simulation_app.is_running():
    my_world.step(render=True)
    # if my_world.is_stopped() and not reset_needed:
    #     reset_needed = True
    # if my_world.is_playing():
    #     if reset_needed:
    #         my_world.reset()
    #         my_controller.reset()
    #         reset_needed = False
    #     if i >= 0 and i < 1000:
    #         # forward
    #         my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
    #         print(my_jetbot.get_linear_velocity())
    #     elif i >= 1000 and i < 1300:
    #         # rotate
    #         my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
    #         print(my_jetbot.get_angular_velocity())
    #     elif i >= 1300 and i < 2000:
    #         # forward
    #         my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
    #     elif i == 2000:
    #         i = 0
    #     i += 1
    if args.test is True:
        break

simulation_app.close()

