"""
approach_node.py — One-shot approach executor.

Activated by the orchestrator via ~/start service.  Reads the latest
filtered target from /wound/pipeline/target, then plans + executes a
MoveGroup trajectory using the multi-stage retry strategy:

  stage 0 :  Pilz PTP   tilt +/-23deg   z_spin +/-23deg
  stage 1 :  Pilz PTP   tilt +/-40deg   z_spin +/-34deg
  stage 2 :  Pilz LIN   tilt +/-40deg   z_spin +/-34deg
  stage 3 :  OMPL        tilt +/-40deg   z_spin +/-57deg

Publishes result on /wound/approach/status:
  "success"  — robot reached target
  "failed"   — all stages exhausted
  "aborted"  — ~/abort called mid-execution

Services:
  ~/start    (std_srvs/Trigger)  — begin approach to latest target
  ~/abort    (std_srvs/Trigger)  — cancel current motion
"""

from typing import NamedTuple

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
MAX_VELOCITY     = 0.15
MAX_ACCELERATION = 0.15
POS_TOLERANCE_M  = 0.01   # 1 cm sphere
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
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='PTP',
           tilt_tol=0.4, z_spin_tol=0.4, plan_time=5.0, attempts=1),
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='PTP',
           tilt_tol=0.7, z_spin_tol=0.6, plan_time=5.0, attempts=1),
    _Stage(pipeline_id=PILZ_PIPELINE, planner_id='LIN',
           tilt_tol=0.7, z_spin_tol=0.6, plan_time=5.0, attempts=1),
    _Stage(pipeline_id=OMPL_PIPELINE, planner_id='',
           tilt_tol=0.7, z_spin_tol=1.0, plan_time=5.0, attempts=5),
]
MAX_STAGE = len(RETRY_STAGES) - 1


class ApproachNode(Node):

    def __init__(self):
        super().__init__('approach_node')

        self._executing    = False
        self._aborting     = False
        self._retry_stage  = 0
        self._target       = None       # latest PoseStamped from orchestrator
        self._goal_handle  = None       # active MoveGroup goal handle

        # MoveGroup action client
        self._client       = ActionClient(self, MoveGroup, '/move_action')
        self._server_ready = False
        self.create_timer(1.0, self._check_server)

        # Target from orchestrator
        self.create_subscription(
            PoseStamped, '/wound/pipeline/target', self._target_cb, 10)

        # Status publisher
        self._pub_status = self.create_publisher(
            String, '/wound/approach/status', 10)

        # Services
        self.create_service(Trigger, '~/start', self._start_cb)
        self.create_service(Trigger, '~/abort', self._abort_cb)

        self.get_logger().info(
            f'Approach node ready  '
            f'[vel={MAX_VELOCITY} acc={MAX_ACCELERATION} '
            f'stages={len(RETRY_STAGES)}: PTP->PTP->LIN->OMPL]')

    def _check_server(self):
        if self._client.server_is_ready():
            if not self._server_ready:
                self._server_ready = True
                self.get_logger().info('/move_action server ready')
        else:
            self.get_logger().info(
                'Waiting for /move_action...', throttle_duration_sec=5.0)

    def _target_cb(self, msg: PoseStamped):
        self._target = msg

    def _start_cb(self, _req, response):
        if self._executing:
            response.success = False
            response.message = 'Already executing — abort first'
            return response

        if not self._server_ready:
            response.success = False
            response.message = '/move_action server not ready'
            return response

        if self._target is None:
            response.success = False
            response.message = 'No target pose received yet'
            return response

        self._executing   = True
        self._aborting    = False
        self._retry_stage = 0

        p = self._target.pose.position
        self.get_logger().info(
            f'Approach started: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}')

        self._send_goal(self._target)

        response.success = True
        response.message = 'Approach started'
        return response

    def _abort_cb(self, _req, response):
        if not self._executing:
            response.success = False
            response.message = 'Not executing'
            return response

        self._aborting = True
        if self._goal_handle is not None:
            self.get_logger().info('Canceling active MoveGroup goal...')
            self._goal_handle.cancel_goal_async()

        response.success = True
        response.message = 'Abort requested'
        return response

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self._pub_status.publish(msg)
        self.get_logger().info(f'Status: {status}')

    def _send_goal(self, target: PoseStamped):
        stage = RETRY_STAGES[self._retry_stage]

        self.get_logger().info(
            f'Planning stage {self._retry_stage}/{MAX_STAGE}: '
            f'{stage.planner_id or "RRTConnect"} '
            f'({stage.pipeline_id.split("/")[-1]})  '
            f'tilt=+/-{np.degrees(stage.tilt_tol):.0f} deg  '
            f'z_spin=+/-{np.degrees(stage.z_spin_tol):.0f} deg')

        # Position constraint: sphere around target
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

        # Orientation constraint
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

        # MotionPlanRequest
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
        goal.planning_options.replan          = False
        goal.planning_options.replan_attempts = 0

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal REJECTED by MoveGroup')
            self._executing   = False
            self._retry_stage = 0
            self._goal_handle = None
            self._publish_status('failed')
            return

        self._goal_handle = handle
        stage = RETRY_STAGES[self._retry_stage]
        self.get_logger().info(
            f'Goal accepted ({stage.planner_id or "RRTConnect"}, '
            f'stage {self._retry_stage}) — executing...')
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._goal_handle = None
        result = future.result().result
        code   = result.error_code.val
        stage  = RETRY_STAGES[self._retry_stage]

        # Aborted mid-execution
        if self._aborting:
            self._executing   = False
            self._aborting    = False
            self._retry_stage = 0
            self._publish_status('aborted')
            return

        # Success
        if code == 1:
            self.get_logger().info(
                f'Approach DONE ({stage.planner_id or "RRTConnect"}, '
                f'stage {self._retry_stage})')
            self._executing   = False
            self._retry_stage = 0
            self._publish_status('success')
            return

        # Retriable: advance to next stage
        if code in (-1, -16):
            if self._retry_stage < MAX_STAGE:
                self._retry_stage += 1
                nxt = RETRY_STAGES[self._retry_stage]
                self.get_logger().warn(
                    f'{code} ({stage.planner_id or "RRTConnect"}) — '
                    f'advancing to stage {self._retry_stage}: '
                    f'{nxt.planner_id or "RRTConnect"} '
                    f'tilt=+/-{np.degrees(nxt.tilt_tol):.0f} deg '
                    f'z_spin=+/-{np.degrees(nxt.z_spin_tol):.0f} deg')
                self._send_goal(self._target)
                return
            else:
                self.get_logger().error(
                    f'Planning failed at max stage ({MAX_STAGE})')
                self._executing   = False
                self._retry_stage = 0
                self._publish_status('failed')
                return

        # Non-retriable error
        labels = {
            -4:  'CONTROL_FAILED',
            -6:  'TIMED_OUT',
            -12: 'GOAL_IN_COLLISION',
            -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
            -14: 'GOAL_CONSTRAINTS_VIOLATED',
        }
        self.get_logger().error(
            f'MoveIt error: {labels.get(code, f"code={code}")}')
        self._executing   = False
        self._retry_stage = 0
        self._publish_status('failed')


def main(args=None):
    rclpy.init(args=args)
    node = ApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
