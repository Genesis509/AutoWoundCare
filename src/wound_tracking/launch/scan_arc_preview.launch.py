"""Preview the chosen arc targets in RViz without any motion.

Requires ur_moveit (move_group) running externally for /compute_ik.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    kinect = Node(
        package='wound_tracking',
        executable='kinect2_driver',
        name='kinect2_driver',
        output='screen',
    )

    calib_tf = Node(
        package='wound_tracking',
        executable='calibrated_tf_publisher',
        name='calibrated_tf_publisher',
        output='screen',
    )

    skel = Node(
        package='wound_tracking',
        executable='skeleton_tracker',
        name='skeleton_tracker',
        output='screen',
    )

    scan = Node(
        package='wound_tracking',
        executable='scan_pose_arc',
        name='scan_pose_arc',
        output='screen',
    )

    return LaunchDescription([kinect, calib_tf, skel, scan])
