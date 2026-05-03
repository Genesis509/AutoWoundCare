"""scan_loop_node — one-shot 3-pose scan sequence executor.

Sequence: INITIAL (dwell in place) → L (move + dwell) → C (move + dwell) → DONE

At the start of each dwell, publishes the pose label on /scan/at_pose so
wound_report_collector can capture wounds for that viewpoint.
After all three poses, publishes 'done' so the collector finalises the report.

Nothing else changed from the original: same Pilz PTP action client,
same joint constraints, same SCAN_PLAN_ONLY env-var behaviour.
"""
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (JointConstraint, Constraints,
                              MotionPlanRequest, PlanningOptions)
from std_msgs.msg import String

PLANNING_GROUP   = 'ur_manipulator'
PILZ_PIPELINE    = 'pilz_industrial_motion_planner'
MAX_VELOCITY     = 0.15
MAX_ACCELERATION = 0.15
JOINT_TOL        = 0.01   # rad
PLAN_ONLY        = os.environ.get('SCAN_PLAN_ONLY', '0') == '1'
DWELL_S          = 5.0    # seconds to hold at each scan pose

# Sequence: index into SEQ_LABELS / SEQ_JIDX
# SEQ_JIDX = None means no move (robot is already here)
# SEQ_JIDX = 0 or 1 means joints[0] or joints[1] from scan_pose_arc
SEQ_LABELS = ['initial', 'L', 'C', 'R']
SEQ_JIDX   = [None,      0,   1,   2]

IDLE     = 'IDLE'
MOVING   = 'MOVING'
DWELLING = 'DWELLING'
DONE     = 'DONE'


class ScanLoopNode(Node):

    def __init__(self):
        super().__init__('scan_loop_node')

        self._joints       = None   # list[list[float]] — one per waypoint
        self._joint_names  = None
        self._seq_pos      = 0      # 0=initial, 1=L, 2=C, ≥3=finished
        self._state        = IDLE
        self._server_ready = False
        self._dwell_start  = 0.0

        self._client      = ActionClient(self, MoveGroup, '/move_action')
        self._pub_at_pose = self.create_publisher(String, '/scan/at_pose', 10)

        self.create_subscription(JointTrajectory, '/scan/target_joints',
                                 self._joints_cb, 10)
        self.create_timer(1.0, self._tick)

        mode = 'PLAN_ONLY' if PLAN_ONLY else 'EXECUTE'
        self.get_logger().info(
            f'Scan sequence [{mode}] initial→L→C — waiting for /scan/target_joints')

    def _joints_cb(self, msg: JointTrajectory):
        if self._joints is not None:
            return
        if len(msg.points) < 3:
            return
        self._joints      = [list(pt.positions) for pt in msg.points]
        self._joint_names = list(msg.joint_names)
        self.get_logger().info('Joint targets received — sequence ready to start')

    def _tick(self):
        if not self._client.server_is_ready():
            self.get_logger().info('Waiting for /move_action…', throttle_duration_sec=5.0)
            return
        if not self._server_ready:
            self._server_ready = True
            self.get_logger().info('/move_action ready')

        if self._state in (MOVING, DONE):
            return
        if self._joints is None:
            return

        if self._state == DWELLING:
            if time.monotonic() - self._dwell_start >= DWELL_S:
                self._seq_pos += 1
                self._state = IDLE
            else:
                return   # still dwelling

        if self._state == IDLE:
            if self._seq_pos >= len(SEQ_LABELS):
                self.get_logger().info('All poses scanned — publishing done')
                self._pub_at_pose.publish(String(data='done'))
                self._state = DONE
                return
            jidx  = SEQ_JIDX[self._seq_pos]
            label = SEQ_LABELS[self._seq_pos]
            if jidx is None:
                self._start_dwell(label)    # initial: already here
            else:
                self._send_goal(jidx, label)

    def _start_dwell(self, label: str):
        self._dwell_start = time.monotonic()
        self._state       = DWELLING
        self._pub_at_pose.publish(String(data=label))
        self.get_logger().info(
            f'At [{label}] — dwelling {DWELL_S:.0f}s for wound scan')

    def _send_goal(self, joint_idx: int, label: str):
        self._state = MOVING

        gc = Constraints()
        for name, val in zip(self._joint_names, self._joints[joint_idx]):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(val)
            jc.tolerance_above = JOINT_TOL
            jc.tolerance_below = JOINT_TOL
            jc.weight          = 1.0
            gc.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.pipeline_id                     = PILZ_PIPELINE
        req.planner_id                      = 'PTP'
        req.group_name                      = PLANNING_GROUP
        req.num_planning_attempts           = 1
        req.allowed_planning_time           = 5.0
        req.max_velocity_scaling_factor     = MAX_VELOCITY
        req.max_acceleration_scaling_factor = MAX_ACCELERATION
        req.goal_constraints.append(gc)

        goal = MoveGroup.Goal()
        goal.request                      = req
        goal.planning_options             = PlanningOptions()
        goal.planning_options.plan_only   = PLAN_ONLY
        goal.planning_options.replan      = False
        goal.planning_options.replan_attempts = 0

        self.get_logger().info(f'→ {label} (joints[{joint_idx}])')
        self._client.send_goal_async(goal).add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        h = future.result()
        if not h.accepted:
            label = SEQ_LABELS[self._seq_pos]
            self.get_logger().error(f'{label}: goal rejected — skipping pose')
            self._seq_pos += 1
            self._state = IDLE
            return
        h.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        code  = future.result().result.error_code.val
        label = SEQ_LABELS[self._seq_pos]
        if code == 1:
            self.get_logger().info(f'{label}: arrived')
            self._start_dwell(label)
        else:
            self.get_logger().warn(f'{label}: move failed (code={code}) — skipping pose')
            self._seq_pos += 1
            self._state = IDLE


def main():
    rclpy.init()
    node = ScanLoopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
