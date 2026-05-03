"""
orchestrator_node.py — Pipeline FSM orchestrator.

Owns the high-level state machine and coordinates detection, approach,
and locking nodes.  All median filtering / buffering lives here so that
downstream executor nodes receive clean, filtered targets.

FSM:
  IDLE  ──(buffer full)──────────> APPROACH
  APPROACH ──(approach success)──> LOCKING
  APPROACH ──(approach failed)───> IDLE
  LOCKING ──(detection lost)─────> IDLE

Subscriptions:
  /wound/target/pose        (PoseStamped)  — raw detections from pose_estimation
  /wound/approach/status    (String)       — result from approach_node
  /wound/locking/status     (String)       — status from locking_node

Publications:
  /wound/pipeline/target    (PoseStamped)  — filtered target for executors
  /wound/pipeline/state     (String)       — FSM state for debug / visualization

Services called:
  /approach_node/start      (Trigger)
  /approach_node/abort      (Trigger)
  /locking_node/start       (Trigger)
  /locking_node/stop        (Trigger)
"""

import collections

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

IDLE     = 'IDLE'
APPROACH = 'APPROACH'
LOCKING  = 'LOCKING'
TARGET_FRAME         = 'base_link'
BUFFER_WINDOW_IDLE   = 7    # poses to collect before triggering approach
BUFFER_WINDOW_LOCK   = 3    # smaller window for responsive locking updates


class OrchestratorNode(Node):

    def __init__(self):
        super().__init__('orchestrator')

        # Separate callback group for service calls so they don't deadlock
        # with the subscription callbacks running on the default group.
        self._srv_cbg = MutuallyExclusiveCallbackGroup()
        self._state = IDLE
        self._buffer = collections.deque(maxlen=BUFFER_WINDOW_IDLE)
        self._pub_target = self.create_publisher(
            PoseStamped, '/wound/pipeline/target', 10)
        self._pub_state = self.create_publisher(
            String, '/wound/pipeline/state', 10)
        self.create_subscription(
            PoseStamped, '/wound/target/pose', self._pose_cb, 10)
        self.create_subscription(
            String, '/wound/approach/status', self._approach_status_cb, 10)
        self.create_subscription(
            String, '/wound/locking/status', self._locking_status_cb, 10)
        self._cli_approach_start = self.create_client(
            Trigger, '/approach_node/start',
            callback_group=self._srv_cbg)
        self._cli_approach_abort = self.create_client(
            Trigger, '/approach_node/abort',
            callback_group=self._srv_cbg)
        self._cli_locking_start = self.create_client(
            Trigger, '/locking_node/start',
            callback_group=self._srv_cbg)
        self._cli_locking_stop = self.create_client(
            Trigger, '/locking_node/stop',
            callback_group=self._srv_cbg)

        self._publish_state()
        self.get_logger().info(
            f'Orchestrator ready  [IDLE -> APPROACH -> LOCKING]')

    def _set_state(self, new_state: str):
        if new_state == self._state:
            return
        self.get_logger().info(f'FSM: {self._state} -> {new_state}')
        self._state = new_state
        self._publish_state()

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self._pub_state.publish(msg)

    def _compute_filtered_pose(self):
        """Compute per-axis position median + hemisphere-aligned quat average."""
        buf = self._buffer

        xs = [p.pose.position.x for p in buf]
        ys = [p.pose.position.y for p in buf]
        zs = [p.pose.position.z for p in buf]

        quats = np.array([[p.pose.orientation.x, p.pose.orientation.y,
                           p.pose.orientation.z, p.pose.orientation.w]
                          for p in buf])
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
        return stable

    def _pose_cb(self, msg: PoseStamped):
        if msg.header.frame_id != TARGET_FRAME:
            return

        if self._state == IDLE:
            self._buffer.append(msg)
            n = len(self._buffer)

            if n < BUFFER_WINDOW_IDLE:
                self.get_logger().info(
                    f'[IDLE] Buffering {n}/{BUFFER_WINDOW_IDLE}',
                    throttle_duration_sec=1.0)
                return

            # Buffer full → compute filtered target, publish, start approach
            stable = self._compute_filtered_pose()
            self._pub_target.publish(stable)
            self._buffer.clear()

            p = stable.pose.position
            self.get_logger().info(
                f'[IDLE] Stable target: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}'
                f' — starting approach')
            self._call_service(self._cli_approach_start)
            self._set_state(APPROACH)

        elif self._state == LOCKING:
            # Use smaller window for responsive tracking
            self._buffer.append(msg)

            if len(self._buffer) < BUFFER_WINDOW_LOCK:
                return

            stable = self._compute_filtered_pose()
            self._pub_target.publish(stable)
            # Don't clear buffer — sliding window via deque maxlen

    # APPROACH state: poses are ignored (approach_node uses the already-
    # published target). We just wait for approach status.

    def _approach_status_cb(self, msg: String):
        if self._state != APPROACH:
            return

        if msg.data == 'success':
            self.get_logger().info(
                '[APPROACH] Success — transitioning to LOCKING')
            # Resize buffer for locking window
            self._buffer = collections.deque(maxlen=BUFFER_WINDOW_LOCK)
            self._call_service(self._cli_locking_start)
            self._set_state(LOCKING)

        elif msg.data in ('failed', 'aborted'):
            self.get_logger().warn(
                f'[APPROACH] {msg.data} — back to IDLE')
            self._buffer = collections.deque(maxlen=BUFFER_WINDOW_IDLE)
            self._set_state(IDLE)

    def _locking_status_cb(self, msg: String):
        if self._state != LOCKING:
            return

        if msg.data == 'lost':
            self.get_logger().warn(
                '[LOCKING] Detection lost — back to IDLE')
            self._buffer = collections.deque(maxlen=BUFFER_WINDOW_IDLE)
            self._set_state(IDLE)

    def _call_service(self, client):
        if not client.service_is_ready():
            self.get_logger().warn(
                f'Service {client.srv_name} not ready — skipping')
            return

        req = Trigger.Request()
        future = client.call_async(req)
        future.add_done_callback(self._service_done_cb)

    def _service_done_cb(self, future):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn(
                    f'Service call returned: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
