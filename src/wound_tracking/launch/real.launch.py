"""
real.launch.py — full real-hardware stack for wound_tracking.

Starts in order:
  1. UR robot driver (requires robot_ip argument)
  2. MoveIt2 move_group for ur16e
  3. calibrated_tf_publisher — reads eye_in_hand.json → publishes tool0 → camera_color_optical_frame
  4. kinect2_driver — ZMQ bridge receiver → ROS2 camera topics
  5. feature_detector, pose_estimation
  6. orchestrator + approach_node + locking_node (pipeline FSM)

Usage:
  ros2 launch wound_tracking real.launch.py robot_ip:=192.168.1.10
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        description='IP address of the UR16e controller',
    )

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_robot_driver'), 'launch', 'ur_control.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type':  'ur16e',
            'robot_ip': LaunchConfiguration('robot_ip'),
        }.items(),
    )

    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type':     'ur16e',
            'launch_rviz': 'true',
        }.items(),
    )
    calibrated_tf = Node(
        package='wound_tracking',
        executable='calibrated_tf_publisher',
        name='calibrated_tf_publisher',
        output='screen',
    )

    kinect2_driver = Node(
        package='wound_tracking',
        executable='kinect2_driver',
        name='kinect2_driver',
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
        robot_ip_arg,
        ur_control,
        ur_moveit,
        calibrated_tf,
        kinect2_driver,
        feature_detector,
        pose_estimation,
        orchestrator,
        approach,
        locking,
    ])
