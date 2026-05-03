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

    yolo_pose = Node(
        package='wound_tracking',
        executable='yolo_pose',
        name='yolo_pose_node',
        output='screen',
    )

    return LaunchDescription([
        kinect2_driver,
        calibrated_tf,
        yolo_pose,
    ])
