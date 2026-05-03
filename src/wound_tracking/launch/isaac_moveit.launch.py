"""
isaac_moveit.launch.py
======================
Launches RViz, MoveIt, Robot State Publisher, and the Isaac Sim trajectory bridge.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue

def generate_launch_description():
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    pkg_wound_tracking = get_package_share_directory('wound_tracking')
    scene_urdf = os.path.join(pkg_wound_tracking, 'urdf', 'scene.urdf.xacro')

    # Dynamically execute xacro with the required arguments passed explicitly
    robot_description_content = Command(
        [FindExecutable(name='xacro'), ' ', scene_urdf, ' name:=ur', ' ur_type:=ur16e']
    )

    # 1. Robot State Publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{
            'robot_description': ParameterValue(robot_description_content, value_type=str),
            'use_sim_time': use_sim_time
        }]
    )

    # 2. Isaac Sim Trajectory Bridge
    bridge_node = Node(
        package='wound_tracking',
        executable='isaac_trajectory_bridge',
        name='isaac_trajectory_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 3. MoveIt2 + RViz
    ur_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': 'ur16e',
            'use_sim_time': use_sim_time,
            'launch_rviz': 'true',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        rsp_node,
        bridge_node,
        ur_moveit
    ])