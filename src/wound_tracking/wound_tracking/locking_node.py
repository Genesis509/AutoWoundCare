"""
locking_node.py — Continuous wound-tracking (lock-on) executor.

Activated by the orchestrator via ~/start.  Continuously consumes filtered
target poses from /wound/pipeline/target and replans small corrections via
Pilz PTP whenever the target drifts beyond a position or orientation
threshold.

Uses tighter tolerances and slower velocity than the approach node to
produce smooth, cable-safe micro-corrections.

Parameters:
  replan_pos_threshold   (m)   — replan if position drift exceeds this
  replan_ori_threshold   (rad) — replan if orientation drift exceeds this
  replan_cooldown        (s)   — minimum gap between replans
  detection_timeout      (s)   — report "lost" if no target for this long
  max_velocity           (float) — velocity scaling factor
  max_acceleration       (float) — acceleration scaling factor

Services:
  ~/start    (std_srvs/Trigger)  — begin tracking
  ~/stop     (std_srvs/Trigger)  — stop tracking

Publishes:
  /wound/locking/status  (std_msgs/String)
    "tracking"  — actively tracking, replanning as needed
    "lost"      — detection lost (timeout)
    "idle"      — not active
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Pose as PoseMsg
from std_msgs.msg import String
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume, Constraints, MotionPlanRequest,
    PlanningOptions, PositionConstraint, OrientationConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger

PLANNING_GROUP   = 'ur_manipulator'
EE_LINK          = 'tool0'
TARGET_FRAME     = 'base_link'
PILZ_PIPELINE    = 'pilz_industrial_motion_planner'
DEFAULT_VELOCITY        = 0.08
DEFAULT_ACCELERATION    = 0.08
DEFAULT_POS_THRESHOLD   = 0.015    # 1.5 cm
DEFAULT_ORI_THRESHOLD   = 0.15     # ~8.6 deg
DEFAULT_COOLDOWN        = 0.5      # seconds
DEFAULT_DETECTION_TMO   = 3.0      # seconds
POS_TOLERANCE_M         = 0.005    # 5 mm sphere (tighter than approach)
TILT_TOLERANCE          = 0.3      # rad
Z_SPIN_TOLERANCE        = 0.3      # rad


class LockingNode(Node):

    def __init__(self):
        super().__init__('locking_node')

        # Parameters
        self.declare_parameter('max_velocity', DEFAULT_VELOCITY)
        self.declare_parameter('max_acceleration', DEFAULT_ACCELERATION)
        self.declare_parameter('replan_pos_threshold', DEFAULT_POS_THRESHOLD)
        self.declare_parameter('replan_ori_threshold', DEFAULT_ORI_THRESHOLD)
        self.declare_parameter('replan_cooldown', DEFAULT_COOLDOWN)
        self.declare_parameter('detection_timeout', DEFAULT_DETECTION_TMO)

        self._vel           = self.get_parameter('max_velocity').value
        self._acc           = self.get_parameter('max_acceleration').value
        self._pos_threshold = self.get_parameter('replan_pos_threshold').value
        self._ori_threshold = self.get_parameter('replan_ori_threshold').value
        self._cooldown      = self.get_parameter('replan_cooldown').value
        self._det_timeout   = self.get_parameter('detection_timeout').value

        # State
        self._active         = False
        self._in_flight      = False     # MoveGroup goal executing
        self._latest_target  = None      # latest PoseStamped from orchestrator
        self._last_sent_pose = None      # last pose we sent to MoveGroup
        self._last_goal_time = None      # time of last dispatched goal
        self._last_target_time = None    # time of last target received

        # MoveGroup client
        self._client       = ActionClient(self, MoveGroup, '/move_action')
        self._server_ready = False
        self.create_timer(1.0, self._check_server)

        # Target from orchestrator
        self.create_subscription(
            PoseStamped, '/wound/pipeline/target', self._target_cb, 10)

        # Status publisher
        self._pub_status = self.create_publisher(
            String, '/wound/locking/status', 10)

        # Services
        self.create_service(Trigger, '~/start', self._start_cb)
        self.create_service(Trigger, '~/stop', self._stop_cb)

        # Periodic tick — check thresholds and detection timeout
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f'Locking node ready  '
            f'[vel={self._vel} pos_thr={self._pos_threshold*1000:.0f}mm '
            f'ori_thr={np.degrees(self._ori_threshold):.0f}deg '
            f'cooldown={self._cooldown}s timeout={self._det_timeout}s]')

    def _check_server(self):
        if self._client.server_is_ready():
            if not self._server_ready:
                self._server_ready = True
                self.get_logger().info('/move_action server ready')
        else:
            self.get_logger().info(
                'Waiting for /move_action...', throttle_duration_sec=5.0)

    def _target_cb(self, msg: PoseStamped):
        self._latest_target = msg
        self._last_target_time = self.get_clock().now()

    def _start_cb(self, _req, response):
        if self._active:
            response.success = True
            response.message = 'Already active'
            return response

        if not self._server_ready:
            response.success = False
            response.message = '/move_action server not ready'
            return response

        self._active         = True
        self._in_flight      = False
        self._last_sent_pose = None
        self._last_goal_time = None
        self._last_target_time = self.get_clock().now()

        self.get_logger().info('Locking STARTED')
        self._publish_status('tracking')

        response.success = True
        response.message = 'Locking started'
        return response

    def _stop_cb(self, _req, response):
        if not self._active:
            response.success = True
            response.message = 'Already idle'
            return response

        self._active = False
        self.get_logger().info('Locking STOPPED')
        self._publish_status('idle')

        response.success = True
        response.message = 'Locking stopped'
        return response

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self._pub_status.publish(msg)

    def _tick(self):
        if not self._active or not self._server_ready:
            return

        now = self.get_clock().now()

        # Detection timeout check
        if self._last_target_time is not None:
            elapsed = (now - self._last_target_time).nanoseconds * 1e-9
            if elapsed > self._det_timeout:
                self.get_logger().warn(
                    f'No target for {elapsed:.1f}s — detection lost')
                self._active = False
                self._publish_status('lost')
                return

        if self._latest_target is None or self._in_flight:
            return

        # Cooldown check
        if self._last_goal_time is not None:
            elapsed = (now - self._last_goal_time).nanoseconds * 1e-9
            if elapsed < self._cooldown:
                return

        # Threshold check: replan only if target drifted enough
        if self._last_sent_pose is not None:
            pos_drift = self._position_distance(
                self._latest_target, self._last_sent_pose)
            ori_drift = self._orientation_distance(
                self._latest_target, self._last_sent_pose)

            if pos_drift < self._pos_threshold and ori_drift < self._ori_threshold:
                return

            self.get_logger().info(
                f'Drift: pos={pos_drift*1000:.1f}mm ori={np.degrees(ori_drift):.1f}deg'
                f' — replanning')
        else:
            self.get_logger().info('Locking: sending first correction')

        self._last_sent_pose = self._latest_target
        self._last_goal_time = now
        self._send_correction(self._latest_target)

    def _position_distance(self, a: PoseStamped, b: PoseStamped) -> float:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        dz = a.pose.position.z - b.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _orientation_distance(self, a: PoseStamped, b: PoseStamped) -> float:
        """Angle between two quaternions (rad)."""
        qa = a.pose.orientation
        qb = b.pose.orientation
        dot = abs(qa.x*qb.x + qa.y*qb.y + qa.z*qb.z + qa.w*qb.w)
        dot = min(dot, 1.0)
        return 2.0 * math.acos(dot)

    def _send_correction(self, target: PoseStamped):
        self._in_flight = True

        # Position constraint: tight sphere
        sphere = SolidPrimitive()
        sphere.type       = SolidPrimitive.SPHERE
        sphere.dimensions = [POS_TOLERANCE_M]

        vol = BoundingVolume()
        vol.primitives.append(sphere)
        center_pose = PoseMsg()
        center_pose.position      = target.pose.position
        center_pose.orientation.w = 1.0
        vol.primitive_poses.append(center_pose)

        pos_con = PositionConstraint()
        pos_con.header            = target.header
        pos_con.link_name         = EE_LINK
        pos_con.weight            = 1.0
        pos_con.constraint_region = vol

        # Orientation constraint: tight
        ori_con = OrientationConstraint()
        ori_con.header                    = target.header
        ori_con.link_name                 = EE_LINK
        ori_con.orientation               = target.pose.orientation
        ori_con.absolute_x_axis_tolerance = TILT_TOLERANCE
        ori_con.absolute_y_axis_tolerance = TILT_TOLERANCE
        ori_con.absolute_z_axis_tolerance = Z_SPIN_TOLERANCE
        ori_con.weight                    = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_con)
        goal_constraints.orientation_constraints.append(ori_con)

        # Pilz PTP only — deterministic, instant planning, minimal joint motion
        req = MotionPlanRequest()
        req.pipeline_id                     = PILZ_PIPELINE
        req.planner_id                      = 'PTP'
        req.group_name                      = PLANNING_GROUP
        req.num_planning_attempts           = 1
        req.allowed_planning_time           = 2.0
        req.max_velocity_scaling_factor     = self._vel
        req.max_acceleration_scaling_factor = self._acc
        req.goal_constraints.append(goal_constraints)

        goal = MoveGroup.Goal()
        goal.request          = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = False
        goal.planning_options.replan_attempts = 0

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Locking correction REJECTED')
            self._in_flight = False
            return
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._in_flight = False
        result = future.result().result
        code   = result.error_code.val

        if code == 1:
            return  # success — silent, will check again on next tick

        labels = {
            -1:  'PLANNING_FAILED',
            -4:  'CONTROL_FAILED',
            -6:  'TIMED_OUT',
            -12: 'GOAL_IN_COLLISION',
            -16: 'INVALID_GOAL_CONSTRAINTS',
        }
        self.get_logger().warn(
            f'Locking correction: {labels.get(code, f"code={code}")} — '
            f'will retry on next tick')
        # Clear last sent so next tick retries immediately
        self._last_sent_pose = None


def main(args=None):
    rclpy.init(args=args)
    node = LockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
