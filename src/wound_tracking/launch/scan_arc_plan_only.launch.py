"""Full arc pipeline with plan-only mode: MoveIt plans each pose and shows the
trajectory in RViz, but the robot does NOT execute.  Requires ur_moveit running.
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

    loop = Node(
        package='wound_tracking',
        executable='scan_loop',
        name='scan_loop_node',
        output='screen',
        additional_env={'SCAN_PLAN_ONLY': '1'},
    )

    return LaunchDescription([kinect, calib_tf, skel, scan, loop])
