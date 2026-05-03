"""
sim_mannequin.launch.py — wound_tracking simulation with humanoid mannequin.

Identical to sim.launch.py except the world file is tracking_scene_mannequin.sdf
which replaces the plain red cube with a humanoid mannequin lying on the table
and a separate red wound patch on the upper chest.

Scene
─────
  - UR16e robot arm standing on the table (base at z = 0.75 m)
  - Humanoid mannequin lying on back, head toward +X
  - Red wound patch on upper chest (world: x=0.53, y=0.0, z=0.895)
    detectable by the existing feature_detector / pose_estimation pipeline

Usage
─────
  ros2 launch wound_tracking sim_mannequin.launch.py

To reposition the mannequin or wound patch without relaunching:
  Edit src/wound_tracking/worlds/tracking_scene_mannequin.sdf
    - "BODY POSITION"  comment block  → whole-body pose
    - "WOUND POSITION" comment block  → wound patch pose
  Then rebuild: colcon build --packages-select wound_tracking
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = get_package_share_directory('wound_tracking')
    scene_urdf = os.path.join(pkg, 'urdf', 'scene.urdf.xacro')
    world_file = os.path.join(pkg, 'worlds', 'tracking_scene_mannequin.sdf')
    #   Custom description_file adds the camera to tool0.
    #   Custom world_file adds the table + mannequin + wound patch.
    #   RViz disabled here — MoveIt launches its own RViz below.
    ur_sim_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_simulation_gz'), 'launch', 'ur_sim_control.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur16e',
            'description_file': scene_urdf,
            'world_file': world_file,
            'launch_rviz': 'false',
        }.items(),
    )

    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur16e',
            'use_sim_time': 'true',
            'launch_rviz': 'true',
        }.items(),
    )

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='camera_bridge',
        arguments=[
            '/wound_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/wound_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/wound_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/wound_camera/image', '/camera/color/image_raw'),
            ('/wound_camera/depth_image', '/camera/depth_registered/image_raw'),
            ('/wound_camera/camera_info', '/camera/color/camera_info'),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    feature_detector = Node(
        package='wound_tracking',
        executable='feature_detector',
        name='feature_detector',
        parameters=[{'use_sim_time': True, 'show_debug_window': True}],
        output='screen',
    )

    pose_estimation = Node(
        package='wound_tracking',
        executable='pose_estimation',
        name='pose_estimation',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    orchestrator = Node(
        package='wound_tracking',
        executable='orchestrator',
        name='orchestrator',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    approach = Node(
        package='wound_tracking',
        executable='approach_node',
        name='approach_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    locking = Node(
        package='wound_tracking',
        executable='locking_node',
        name='locking_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    processing_nodes = TimerAction(
        period=8.0,
        actions=[camera_bridge, feature_detector, pose_estimation],
    )

    view_detection = Node(
        package='wound_tracking',
        executable='view_detection',
        name='view_detection',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        ur_sim_control,
        ur_moveit,
        processing_nodes,
        orchestrator,
        approach,
        locking,
        view_detection,
    ])
