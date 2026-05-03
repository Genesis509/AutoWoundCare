"""
sim_motion.launch.py — motion-only Gazebo simulation.

Stripped copy of sim_mannequin.launch.py: keeps Gz + UR16e + MoveIt + RViz +
the same world (mannequin lying on table), drops all perception (camera bridge,
yolo, feature_detector, pose_estimation, orchestrator, approach, locking,
view_detection).

Replaces them with:
  - fake_scan_targets : hardcoded 3 PoseStamped on /scan/target_poses (base_link)
  - scan_loop         : sends each pose to /move_action sequentially, loops

Robot home pose is set via the URDF `initial_positions` block in
scene.urdf.xacro — Gz spawns the arm directly at home, so MoveIt's initial
state equals home.

Mannequin pose lives in worlds/tracking_scene_mannequin.sdf
(BODY POSITION block).

Bridge to real:
  swap fake_scan_targets for kinect2_driver + calibrated_tf_publisher +
  skeleton_tracker + scan_pose_publisher.  scan_loop and motion path unchanged.
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
    #   ur_sim_control spawns Gazebo Gz with the custom scene URDF (camera on
    #   tool0, robot base at z=0.75) and the mannequin world.
    #   Robot joints initialize at the URDF `initial_positions` (= home pose).
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
    #   Delayed so Gz, controllers, and MoveIt's /move_action server are up.
    fake_targets = Node(
        package='wound_tracking',
        executable='fake_scan_targets',
        name='fake_scan_targets_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    scan_loop = Node(
        package='wound_tracking',
        executable='scan_loop',
        name='scan_loop_node',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    motion_nodes = TimerAction(period=10.0, actions=[fake_targets, scan_loop])

    return LaunchDescription([ur_sim_control, ur_moveit, motion_nodes])
