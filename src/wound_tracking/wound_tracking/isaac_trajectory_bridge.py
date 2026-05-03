#!/usr/bin/env python3
"""
isaac_trajectory_bridge.py
==========================
Receives FollowJointTrajectory actions from MoveIt2 and streams them as
sensor_msgs/JointState to Isaac Sim via the /joint_command topic.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

class IsaacTrajectoryBridge(Node):
    def __init__(self):
        super().__init__('isaac_trajectory_bridge')
        
        # Publisher to Isaac Sim
        self._cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        
        # Action server for MoveIt
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_callback
        )
        self.get_logger().info("Isaac Trajectory Bridge ready.")

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing MoveIt trajectory...')
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        
        start_time = self.get_clock().now().nanoseconds / 1e9

        for point in trajectory.points:
            # Calculate absolute target time
            sec = point.time_from_start.sec
            nanosec = point.time_from_start.nanosec
            target_time_from_start = sec + (nanosec / 1e9)
            
            # Wait until it is time to publish this point
            while True:
                current_time = self.get_clock().now().nanoseconds / 1e9
                elapsed = current_time - start_time
                if elapsed >= target_time_from_start:
                    break
                time.sleep(0.005) # Sleep 5ms to prevent CPU pegging
            
            # Build and send JointState command
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = list(point.positions)
            if point.velocities:
                msg.velocity = list(point.velocities)
            if point.effort:
                msg.effort = list(point.effort)
                
            self._cmd_pub.publish(msg)

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        self.get_logger().info('Trajectory execution complete.')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = IsaacTrajectoryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()