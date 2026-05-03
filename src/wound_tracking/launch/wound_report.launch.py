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

    skeleton_tracker = Node(
        package='wound_tracking',
        executable='skeleton_tracker',
        name='skeleton_tracker',
        output='screen',
    )

    yolo_pose = Node(
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
        kinect2_driver,
        calibrated_tf,
        skeleton_tracker,
        yolo_pose,
        collector,
    ])
