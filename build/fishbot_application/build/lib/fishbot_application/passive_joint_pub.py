#!/usr/bin/env python3
"""
合并关节状态：桥接的 Gazebo joint_states + 被动关节，发布到 /joint_states。

关键改进：使用定时器定时发布，即使 Gazebo bridge 不工作，
也能确保 robot_state_publisher 收到所有关节的默认值，使轮子在 RViz 中可见。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateMerger(Node):
    def __init__(self):
        super().__init__('joint_state_merger')

        # 所有非 fixed 关节的默认值（包括被动关节 + 驱动轮）
        # 即使 bridge 不工作，这些默认值也能让 robot_state_publisher 生成轮子的 TF
        self.joint_positions = {
            'front_turn_joint': 0.0,
            'front_roll_joint': 0.0,
            'left_joint': 0.0,
            'right_joint': 0.0,
        }

        # 接收 bridge 转发的 Gazebo 关节数据（会在定时器发布之间更新实际值）
        self.sub = self.create_subscription(
            JointState,
            '/world/default/model/mini_diff_robot/joint_state',
            self.bridge_callback,
            10)

        # 发布合并后的完整 joint_state
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # 定时器：无论 bridge 是否工作，定期发布 joint_states
        self._timer = self.create_timer(0.1, self._timer_publish)

        self.get_logger().info('joint_state_merger started with timer-based publishing')

    def bridge_callback(self, msg: JointState):
        """bridge 数据到达时更新关节位置"""
        for i, name in enumerate(msg.name):
            if name in self.joint_positions and i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

    def _timer_publish(self):
        """定时发布完整的 joint_states"""
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self.joint_positions.keys())
        out.position = list(self.joint_positions.values())
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(JointStateMerger())


if __name__ == '__main__':
    main()
