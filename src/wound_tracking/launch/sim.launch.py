"""
sim.launch.py — full simulation stack for wound_tracking.

Includes the two upstream UR launch files separately (instead of the combined
ur_sim_moveit.launch.py) so we can pass both description_file and world_file:

  1. ur_sim_control.launch.py  — Gz + UR16e ros2_control (custom URDF + world)
  2. ur_moveit.launch.py       — MoveIt2 move_group + RViz

On top of that:
  3. ros_gz_bridge for camera topics (Gz -> ROS2)
  4. feature_detector, pose_estimation (delayed 8 s)
  5. orchestrator + approach_node + locking_node (pipeline FSM)

Usage:
  ros2 launch wound_tracking sim.launch.py
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
    world_file = os.path.join(pkg, 'worlds', 'tracking_scene.sdf')
    #   Custom description_file adds the camera to tool0.
    #   Custom world_file adds the table + red target cube.
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
