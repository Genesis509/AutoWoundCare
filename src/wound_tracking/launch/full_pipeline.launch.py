    """full_pipeline.launch.py — complete scan pipeline.

Starts all nodes needed for an autonomous 3-pose wound scan:
  kinect2_driver        → camera streams
  calibrated_tf_publisher → camera↔robot TF
  skeleton_tracker      → MediaPipe pose on /skeleton/landmarks
  scan_pose_arc         → computes L/C viewpoints from skeleton, publishes joint targets
  scan_loop             → executes initial→L→C sequence, publishes /scan/at_pose
  yolo_pose             → YOLO wound detection + 3D pose
  wound_report_collector → captures at each pose, deduplicates, writes report
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

    scan_arc = Node(
        package='wound_tracking',
        executable='scan_pose_arc',
        name='scan_pose_arc',
        output='screen',
    )

    scan_loop = Node(
        package='wound_tracking',
        executable='scan_loop',
        name='scan_loop_node',
        output='screen',
    )

    yolo = Node(
        package='wound_tracking',
        executable='yolo_pose',
        name='yolo_pose_node',
        output='screen',
    )

    collector = Node(
        package='wound_tracking',
        executable='wound_report_collector',
        name='wound_report_collector',
        output='screen',
    )

    return LaunchDescription([
        kinect,
        calib_tf,
        skel,
        scan_arc,
        scan_loop,
        yolo,
        collector,
    ])
