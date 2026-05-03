"""
real_mannequin.launch.py — Real-hardware locking pipeline with YOLO wound detection.

Assumes UR driver and MoveIt are already running in separate terminals:
  Terminal 1: ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur16e robot_ip:=<IP>
  Terminal 2: ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur16e launch_rviz:=true

Pipeline:
  Kinect2 → yolo_wound_detector → /wound/target/pose
                                       ↓
             orchestrator FSM (IDLE → APPROACH → LOCKING)
                ↙                          ↘
        approach_node                 locking_node

Usage:
  ros2 launch wound_tracking real_mannequin.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    kinect2_driver = Node(
        package='wound_tracking',
        executable='kinect2_driver',
        name='kinect2_driver',
        output='screen',
    )

    calibrated_tf = Node(
        package='wound_tracking',
        executable='calibrated_tf_publisher',
        name='calibrated_tf_publisher',
        output='screen',
    )

    yolo_wound_detector = Node(
        package='wound_tracking',
        executable='yolo_wound_detector',
        name='yolo_wound_detector',
        output='screen',
    )

    orchestrator = Node(
        package='wound_tracking',
        executable='orchestrator',
        name='orchestrator',
        output='screen',
    )

    approach = Node(
        package='wound_tracking',
        executable='approach_node',
        name='approach_node',
        output='screen',
    )

    locking = Node(
        package='wound_tracking',
        executable='locking_node',
        name='locking_node',
        output='screen',
    )

    return LaunchDescription([
        kinect2_driver,
        calibrated_tf,
        yolo_wound_detector,
        orchestrator,
        approach,
        locking,
    ])
