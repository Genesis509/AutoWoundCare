"""
isaac_ros2_setup.py — Isaac Sim 5.1 ROS2 OmniGraph setup
=========================================================
"""

import omni.graph.core as og
import omni.kit.app
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, create_prim
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    print("[setup] isaacsim.ros2.bridge enabled.")
else:
    print("[setup] isaacsim.ros2.bridge already enabled.")
CAMERA_PRIM = "/World/ur16e/wrist_3_link/CameraMount/RGBCamera"
ROBOT_PRIM  = "/World/ur16e"
FRAME_ID    = "camera_color_optical_frame"

CAMERA_GRAPH_PATH = "/World/Graphs/CameraGraph"
ROBOT_GRAPH_PATH = "/World/Graphs/RobotGraph"
for path in [CAMERA_GRAPH_PATH, ROBOT_GRAPH_PATH]:
    if get_prim_at_path(path):
        delete_prim(path)
        print(f"[setup] Deleted existing graph at {path} for clean rebuild.")

if not get_prim_at_path("/World/Graphs"):
    create_prim("/World/Graphs", "Scope")

keys = og.Controller.Keys
(camera_graph, camera_nodes, _, _) = og.Controller.edit(
    {"graph_path": CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick",     "omni.graph.action.OnPlaybackTick"),
            ("RenderProd", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("PubRGB",     "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("PubDepth",   "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("PubCamInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ],
        keys.SET_VALUES: [
            ("RenderProd.inputs:cameraPrim",    CAMERA_PRIM),
            ("RenderProd.inputs:enabled",       True),

            ("PubRGB.inputs:type",              "rgb"),
            ("PubRGB.inputs:topicName",         "camera/color/image_raw"),
            ("PubRGB.inputs:frameId",           FRAME_ID),

            ("PubDepth.inputs:type",            "depth"),
            ("PubDepth.inputs:topicName",       "camera/depth/image_raw"),
            ("PubDepth.inputs:frameId",         FRAME_ID),

            ("PubCamInfo.inputs:topicName",     "camera/color/camera_info"),
            ("PubCamInfo.inputs:frameId",       FRAME_ID),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick",                      "RenderProd.inputs:execIn"),
            ("RenderProd.outputs:execOut",               "PubRGB.inputs:execIn"),
            ("RenderProd.outputs:execOut",               "PubDepth.inputs:execIn"),
            ("RenderProd.outputs:execOut",               "PubCamInfo.inputs:execIn"),
            ("RenderProd.outputs:renderProductPath",     "PubRGB.inputs:renderProductPath"),
            ("RenderProd.outputs:renderProductPath",     "PubDepth.inputs:renderProductPath"),
            ("RenderProd.outputs:renderProductPath",     "PubCamInfo.inputs:renderProductPath"),
        ],
    }
)
(robot_graph, robot_nodes, _, _) = og.Controller.edit(
    {"graph_path": ROBOT_GRAPH_PATH, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick",     "omni.graph.action.OnPlaybackTick"),
            ("ReadTime",   "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("PubJS",      "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubJC",      "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArtCtrl",    "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.SET_VALUES: [
            ("PubJS.inputs:targetPrim",          ROBOT_PRIM),
            ("PubJS.inputs:topicName",           "joint_states"),
            ("SubJC.inputs:topicName",           "joint_command"),
            ("ArtCtrl.inputs:robotPath",         ROBOT_PRIM),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick",                     "PubJS.inputs:execIn"),
            ("OnTick.outputs:tick",                     "SubJC.inputs:execIn"),
            ("OnTick.outputs:tick",                     "ArtCtrl.inputs:execIn"),

            ("ReadTime.outputs:simulationTime",         "PubJS.inputs:timeStamp"),

            ("SubJC.outputs:jointNames",                "ArtCtrl.inputs:jointNames"),
            ("SubJC.outputs:positionCommand",           "ArtCtrl.inputs:positionCommand"),
            ("SubJC.outputs:velocityCommand",           "ArtCtrl.inputs:velocityCommand"),
            ("SubJC.outputs:effortCommand",             "ArtCtrl.inputs:effortCommand"),
        ],
    }
)
print("\n[setup] Done. UR16e Action graphs rebuilt successfully.")
print("  /World/Graphs/CameraGraph  — RGB + depth + camera_info")
print("  /World/Graphs/RobotGraph   — joint_states + joint_command")
print("  Press PLAY to stream topics to ROS2 Jazzy on WSL2.")