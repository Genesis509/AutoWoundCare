"""
pose_detection.launch.py

Launches the perception pipeline only — no motion, no MoveIt.
Run ur_robot_driver separately before this.

  kinect2_driver          — ZMQ bridge → ROS2 camera topics
  calibrated_tf_publisher — eye_in_hand.json → TF tool0 → camera_color_optical_frame
  feature_detector        — HSV red detection → bbox + centroid
  pose_estimation         — depth + TF → 3D pose + RViz markers

Usage:
  ros2 launch wound_tracking pose_detection.launch.py
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

    feature_detector = Node(
        package='wound_tracking',
        executable='feature_detector',
        name='feature_detector',
        output='screen',
    )

    pose_estimation = Node(
        package='wound_tracking',
        executable='pose_estimation',
        name='pose_estimation',
        output='screen',
    )

    return LaunchDescription([
        kinect2_driver,
        calibrated_tf,
        feature_detector,
        pose_estimation,
    ])
