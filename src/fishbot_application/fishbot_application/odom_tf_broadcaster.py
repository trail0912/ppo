#!/usr/bin/env python3
"""
odom -> base_footprint TF 发布器 + /tf 时间戳统一修正 + 雷达时间戳修正。

epoch offset 应用到所有 /tf 消息和 /scan，保证整棵 TF 树时间戳一致。
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time as TimeMsg


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.tf_broadcaster = TransformBroadcaster(self)

        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_z = 0.0
        self._offset_set = False
        self._last_sim_ns = 0
        self._epoch_offset_ns = 0
        self._msg_count = 0

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.reset_sub = self.create_subscription(Bool, '/robot_reset', self.reset_callback, 10)

        # 雷达时间戳修正
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # /tf 时间戳统一修正：所有非 odom→base_footprint 的 tf 也加 epoch offset
        self.tf_sub = self.create_subscription(TFMessage, '/tf_uncorrected', self.tf_callback, 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        self._timer = self.create_timer(0.05, self._timer_tick)

        self.get_logger().info('odom_tf_broadcaster ready')

    # ── 定时器 ──────────────────────────────────────────
    def _timer_tick(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._offset_set and self._last_sim_ns > 1_000_000_000 and now_ns < self._last_sim_ns:
            self.get_logger().warn(
                f'SIM TIME JUMP {self._last_sim_ns/1e9:.1f}→{now_ns/1e9:.1f} '
                f'(epoch += {self._last_sim_ns/1e9:.1f}s)')
            self._epoch_offset_ns += self._last_sim_ns
            self._offset_set = False
        self._last_sim_ns = now_ns
        if not self._offset_set:
            self._send_tf(0, 0, 0, 0, 0, 0, 1)

    # ── odometry → TF (odom→base_footprint) ─────────────
    def odom_callback(self, msg: Odometry):
        if not self._offset_set:
            self._offset_x = msg.pose.pose.position.x
            self._offset_y = msg.pose.pose.position.y
            self._offset_z = msg.pose.pose.position.z
            self._offset_set = True
            self.get_logger().info(f'OFFSET=({self._offset_x:.2f},{self._offset_y:.2f}) '
                                   f'epoch={self._epoch_offset_ns/1e9:.1f}s')

        x = msg.pose.pose.position.x - self._offset_x
        y = msg.pose.pose.position.y - self._offset_y
        self._msg_count += 1
        if self._msg_count % 30 == 0:
            self.get_logger().info(f'TF=({x:.2f},{y:.2f})')

        t = TransformStamped()
        t.header.stamp = self._fix_stamp(msg.header.stamp)
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = msg.pose.pose.position.z - self._offset_z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # ── 雷达时间戳修正 ─────────────────────────────────
    def scan_callback(self, msg: LaserScan):
        corrected = LaserScan()
        corrected.header.stamp = self._fix_stamp(msg.header.stamp)
        corrected.header.frame_id = msg.header.frame_id
        corrected.angle_min = msg.angle_min
        corrected.angle_max = msg.angle_max
        corrected.angle_increment = msg.angle_increment
        corrected.time_increment = msg.time_increment
        corrected.scan_time = msg.scan_time
        corrected.range_min = msg.range_min
        corrected.range_max = msg.range_max
        corrected.ranges = msg.ranges
        corrected.intensities = msg.intensities
        self.scan_pub.publish(corrected)

    # ── /tf 统一修正（robot_state_publisher 的 tf 也加 epoch offset）──
    def tf_callback(self, msg: TFMessage):
        corrected = TFMessage()
        for t in msg.transforms:
            ct = TransformStamped()
            ct.header.stamp = self._fix_stamp(t.header.stamp)
            ct.header.frame_id = t.header.frame_id
            ct.child_frame_id = t.child_frame_id
            ct.transform = t.transform
            corrected.transforms.append(ct)
        self.tf_pub.publish(corrected)

    # ── reset 信号 ───────────────────────────────────────
    def reset_callback(self, msg: Bool):
        self.get_logger().warn(f'/robot_reset (epoch={self._epoch_offset_ns/1e9:.1f}s)')
        self._offset_set = False
        self._send_tf(0, 0, 0, 0, 0, 0, 1)

    # ── helpers ──────────────────────────────────────────
    def _fix_stamp(self, stamp):
        total_ns = stamp.sec * 1_000_000_000 + stamp.nanosec + self._epoch_offset_ns
        result = TimeMsg()
        result.sec = int(total_ns // 1_000_000_000)
        result.nanosec = int(total_ns % 1_000_000_000)
        return result

    def _send_tf(self, x, y, z, qx, qy, qz, qw):
        t = TransformStamped()
        t.header.stamp = self._fix_stamp(self.get_clock().now().to_msg())
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomTfBroadcaster())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
