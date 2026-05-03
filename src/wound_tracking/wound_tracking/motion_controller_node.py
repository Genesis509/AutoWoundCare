"""
motion_controller_node.py — SIM and REAL.

One-shot motion with full 6-DOF targeting + adaptive wrist-3 locking.

Collects MEDIAN_WINDOW target poses, computes a per-axis position median and a
hemisphere-aligned quaternion average, then sends a MoveGroup plan+execute goal.

Planning strategy — uses three MoveIt planning pipelines in priority order:

  Pilz PTP  (point-to-point):  Deterministic, shortest joint-space interpolation.
             No sampling, no randomness, no wide arcs.  Instant planning (~ms).
             All joints move linearly from start to goal → wrist_3 barely moves.

  Pilz LIN  (linear Cartesian): Straight-line motion in task space.  Tool moves
             along a ruler-straight 3D path.  Can fail near singularities.

  OMPL RRTConnect (fallback):   Sampling-based.  Finds *a* solution, not the
             shortest.  Only used when Pilz cannot solve.  Wrist-3 path
             constraint added here to prevent mid-trajectory camera spin.

Retry stages:
  stage 0 :  Pilz PTP   tilt ±23°   z_spin ±23°             ← tightest, direct
  stage 1 :  Pilz PTP   tilt ±40°   z_spin ±34°             ← wider tilt
  stage 2 :  Pilz LIN   tilt ±40°   z_spin ±34°             ← Cartesian line
  stage 3 :  OMPL        tilt ±40°   z_spin ±57°             ← last resort

State machine:
  IDLE  → buffer full                          → MOVING
  MOVING → success                             → DONE
  MOVING → PLANNING_FAILED, stage < max        → next stage → MOVING
  MOVING → PLANNING_FAILED, stage == max       → IDLE
  MOVING → other MoveIt error                  → IDLE
  DONE  → ~/reset service                      → IDLE
"""

import collections
from typing import NamedTuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Pose as PoseMsg
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
MAX_VELOCITY     = 0.15
MAX_ACCELERATION = 0.15
POS_TOLERANCE_M  = 0.01   # 1 cm sphere

MEDIAN_WINDOW    = 7
PILZ_PIPELINE = 'pilz_industrial_motion_planner'
OMPL_PIPELINE = 'ompl'


class _Stage(NamedTuple):
    pipeline_id: str
    planner_id:  str
    tilt_tol:    float
    z_spin_tol:  float
    plan_time:   float
    attempts:    int

RETRY_STAGES = [
    # Stage 0: Pilz PTP — direct joint interpolation, tight tilt, tight spin
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='PTP',
           tilt_tol=0.4, z_spin_tol=0.4, plan_time=5.0, attempts=1),

    # Stage 1: Pilz PTP — wider tilt, moderate spin
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='PTP',
           tilt_tol=0.7, z_spin_tol=0.6, plan_time=5.0, attempts=1),

    # Stage 2: Pilz LIN — Cartesian straight line, moderate spin
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='LIN',
           tilt_tol=0.7, z_spin_tol=0.6, plan_time=5.0, attempts=1),

    # Stage 3: OMPL RRTConnect — wider spin as last resort
    _Stage(pipeline_id=OMPL_PIPELINE, planner_id='',
           tilt_tol=0.7, z_spin_tol=1.0, plan_time=5.0, attempts=5),
]
MAX_STAGE = len(RETRY_STAGES) - 1

IDLE   = 0
MOVING = 1
DONE   = 2


class MotionControllerNode(Node):

    def __init__(self):
        super().__init__('motion_controller')

        self._state        = IDLE
        self._buffer       = collections.deque(maxlen=MEDIAN_WINDOW)
        self._last_target  = None
        self._retry_stage  = 0      # index into RETRY_STAGES

        self._client       = ActionClient(self, MoveGroup, '/move_action')
        self._server_ready = False
        self.create_timer(1.0, self._check_server)

        self.create_subscription(PoseStamped, '/wound/target/pose', self._pose_cb, 10)
        self.create_service(Trigger, '~/reset', self._reset_cb)

        self.get_logger().info(
            f'Motion controller ready  '
            f'[window={MEDIAN_WINDOW} vel={MAX_VELOCITY} acc={MAX_ACCELERATION} '
            f'stages={len(RETRY_STAGES)}: PTP→PTP→LIN→OMPL]')

    def _check_server(self):
        if self._client.server_is_ready():
            if not self._server_ready:
                self._server_ready = True
                self.get_logger().info('/move_action server ready')
        else:
            self.get_logger().info('Waiting for /move_action…', throttle_duration_sec=5.0)

    def _reset_cb(self, _req, response):
        self._state       = IDLE
        self._retry_stage = 0
        self._buffer.clear()
        response.success = True
        response.message = 'Reset to IDLE — will plan on next stable detection'
        self.get_logger().info('Reset to IDLE')
        return response

    def _pose_cb(self, msg: PoseStamped):
        if not self._server_ready or self._state != IDLE:
            return

        if msg.header.frame_id != TARGET_FRAME:
            self.get_logger().warn(
                f'Expected frame {TARGET_FRAME}, got {msg.header.frame_id}',
                throttle_duration_sec=5.0)
            return

        self._buffer.append(msg)
        n = len(self._buffer)

        if n < MEDIAN_WINDOW:
            self.get_logger().info(f'Buffering {n}/{MEDIAN_WINDOW}', throttle_duration_sec=1.0)
            return

        xs = [p.pose.position.x for p in self._buffer]
        ys = [p.pose.position.y for p in self._buffer]
        zs = [p.pose.position.z for p in self._buffer]

        quats = np.array([[p.pose.orientation.x, p.pose.orientation.y,
                           p.pose.orientation.z, p.pose.orientation.w]
                          for p in self._buffer])
        q0 = quats[0]
        for i in range(1, len(quats)):
            if np.dot(quats[i], q0) < 0.0:
                quats[i] *= -1.0
        avg_q = quats.mean(axis=0)
        avg_q /= np.linalg.norm(avg_q)

        stable = PoseStamped()
        stable.header.frame_id    = TARGET_FRAME
        stable.header.stamp       = self.get_clock().now().to_msg()
        stable.pose.position.x    = float(np.median(xs))
        stable.pose.position.y    = float(np.median(ys))
        stable.pose.position.z    = float(np.median(zs))
        stable.pose.orientation.x = float(avg_q[0])
        stable.pose.orientation.y = float(avg_q[1])
        stable.pose.orientation.z = float(avg_q[2])
        stable.pose.orientation.w = float(avg_q[3])

        p = stable.pose.position
        self.get_logger().info(
            f'Stable target: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} — sending goal')

        self._state       = MOVING
        self._retry_stage = 0
        self._buffer.clear()
        self._send_goal(stable)

    def _send_goal(self, target: PoseStamped):
        self._last_target = target
        stage = RETRY_STAGES[self._retry_stage]

        self.get_logger().info(
            f'Planning stage {self._retry_stage}/{MAX_STAGE}: '
            f'{stage.planner_id or "RRTConnect"} ({stage.pipeline_id.split("/")[-1]})  '
            f'tilt=±{np.degrees(stage.tilt_tol):.0f}°  z_spin=±{np.degrees(stage.z_spin_tol):.0f}°')

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

        ori_con = OrientationConstraint()
        ori_con.header                    = target.header
        ori_con.link_name                 = EE_LINK
        ori_con.orientation               = target.pose.orientation
        ori_con.absolute_x_axis_tolerance = stage.tilt_tol
        ori_con.absolute_y_axis_tolerance = stage.tilt_tol
        ori_con.absolute_z_axis_tolerance = stage.z_spin_tol
        ori_con.weight                    = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_con)
        goal_constraints.orientation_constraints.append(ori_con)

        req = MotionPlanRequest()
        req.pipeline_id                     = stage.pipeline_id
        req.planner_id                      = stage.planner_id
        req.group_name                      = PLANNING_GROUP
        req.num_planning_attempts           = stage.attempts
        req.allowed_planning_time           = stage.plan_time
        req.max_velocity_scaling_factor     = MAX_VELOCITY
        req.max_acceleration_scaling_factor = MAX_ACCELERATION
        req.goal_constraints.append(goal_constraints)

        goal = MoveGroup.Goal()
        goal.request          = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = False  # we handle retries ourselves
        goal.planning_options.replan_attempts = 0

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal REJECTED by MoveGroup')
            self._retry_stage = 0
            self._state = IDLE
            return
        stage = RETRY_STAGES[self._retry_stage]
        self.get_logger().info(
            f'Goal accepted ({stage.planner_id or "RRTConnect"}, '
            f'stage {self._retry_stage}) — executing…')
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        code   = result.error_code.val
        stage  = RETRY_STAGES[self._retry_stage]

        if code == 1:
            self.get_logger().info(
                f'Motion DONE ({stage.planner_id or "RRTConnect"}, '
                f'stage {self._retry_stage}) — robot holding at target')
            self._retry_stage = 0
            self._state = DONE
            return

        # Retriable errors: advance to the next planning stage
        #   -1  PLANNING_FAILED
        #   -16 INVALID_GOAL_CONSTRAINTS (e.g. Pilz rejects mixed constraints)
        if code in (-1, -16):
            if self._retry_stage < MAX_STAGE:
                self._retry_stage += 1
                nxt = RETRY_STAGES[self._retry_stage]
                self.get_logger().warn(
                    f'{code} ({stage.planner_id or "RRTConnect"}) — '
                    f'advancing to stage {self._retry_stage}: '
                    f'{nxt.planner_id or "RRTConnect"} '
                    f'tilt=±{np.degrees(nxt.tilt_tol):.0f}° z_spin=±{np.degrees(nxt.z_spin_tol):.0f}°')
                self._send_goal(self._last_target)
                return
            else:
                self.get_logger().error(
                    f'Planning failed at max stage ({MAX_STAGE}) — back to IDLE')
                self._retry_stage = 0
                self._state = IDLE
                return

        labels = {
            -4:  'CONTROL_FAILED',
            -6:  'TIMED_OUT',
            -12: 'GOAL_IN_COLLISION',
            -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            -14: 'GOAL_CONSTRAINTS_VIOLATED',
        }
        self.get_logger().error(
            f'MoveIt error: {labels.get(code, f"code={code}")} — back to IDLE')
        self._retry_stage = 0
        self._state = IDLE


def main(args=None):
    rclpy.init(args=args)
    node = MotionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
