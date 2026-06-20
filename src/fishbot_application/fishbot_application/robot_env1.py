import random
import time
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, PoseStamped, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from std_msgs.msg import ColorRGBA, Bool
import math
from threading import Lock
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from ros_gz_interfaces.msg import WorldControl, Entity
import subprocess

import os
import csv
from datetime import datetime

TIME_USE = 0.0
arrive = 0
wall = 0
timeout = 0


class RobotEnv1(gym.Env):
    """PPO 训练环境（平坦观测 363 维），使用 Gazebo Ignition API"""

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        super(RobotEnv1, self).__init__()
        self.node = rclpy.create_node('robot_env1')

        # 传感器和执行器
        self.laser_sub = self.node.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # 可视化发布者
        self.robot_markers_pub = self.node.create_publisher(MarkerArray, '/training_visualization', 10)
        self.trajectory_pub = self.node.create_publisher(Marker, '/robot_trajectory', 10)
        self.goal_pub = self.node.create_publisher(Marker, '/training_goal', 10)

        # TF 由 launch 文件的 odom_tf_broadcaster 统一发布
        self.reset_signal_pub = self.node.create_publisher(Bool, '/robot_reset', 10)

        # Gazebo Ignition 服务
        self.world_control_client = self.node.create_client(
            ControlWorld, '/world/default/control')
        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/world/default/control not available, waiting...')

        self.set_pose_client = self.node.create_client(
            SetEntityPose, '/world/default/set_pose')
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/world/default/set_pose not available, waiting...')

        # 确保初始位姿有效
        self.initial_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.robot_pose = self.initial_pose.copy()

        # odom 偏移
        self._odom_offset_set = False

        # 数据存储
        self.laser_data = np.zeros(360)
        self.robot_path = []
        self.current_goal = None
        self.path_lock = Lock()
        self.step_count = 0
        self._is_terminated = False
        self._last_collision_time = 0
        self._resetting = False  # reset 期间抑制轨迹发布
        self._episode_count = 0  # 当前模型内的回合计数
        self._episode_stats = []  # 每回合统计 [{event, reward, steps, goal_dist}]
        self.goal = 0
        self.goal_dist = 0
        self.reward = 0
        self.angle_to_goal = 0
        self.angle = 0
        self.zhuang = False
        self.min_distance = 0
        self.new_time = 0
        self.distance_reward = 0
        self.obstacle_penalty = 0

        self.spawn_robot_cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-topic', '/robot_description',
            '-name', 'mini_diff_robot'
        ]

        # 动作和观测空间（平坦 363 维：360 激光 + 3 位姿）
        self.action_space = gym.spaces.Discrete(4)  # 0:前进, 1:左转, 2:右转, 3:右转
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * 360 + [-10.0, -10.0, -np.pi], dtype=np.float32),
            high=np.array([10.0] * 360 + [10.0, 10.0, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        # 初始化日志文件
        self.log_dir = "train_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = os.path.join(self.log_dir, f"navigation_events_{time_str}.csv")
        with open(self.event_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_type", "step_count", "episode_reward",
                "robot_x", "robot_y", "goal_x", "goal_y", "goal_dist"
            ])

    def reset_episode_counter(self):
        """每个新模型开始时调用，回合计数从 1 开始"""
        self._episode_count = 0

    def reset_stats(self):
        """每个新模型开始时调用，清空回合统计"""
        self._episode_stats = []

    def get_stats(self):
        """获取当前模型所有回合的统计"""
        return self._episode_stats.copy()

    def set_goal(self, x, y):
        """设置训练目标点"""
        self.current_goal = (x, y)
        self.goal = math.hypot(x - self.robot_pose['x'], y - self.robot_pose['y'])
        self.goal_dist = self.goal
        self._publish_visualization()

    def reset_robot_pose(self, x=0.0, y=0.0, yaw=0.0):
        """强制设置 Gazebo 中机器人位姿"""
        req = SetEntityPose.Request()
        req.entity.name = 'mini_diff_robot'
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = 0.05
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = math.sin(yaw / 2.0)
        req.pose.orientation.w = math.cos(yaw / 2.0)
        future = self.set_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.node.get_logger().warn("set_pose service call failed or timeout")
            return False
        return True

    def _robot_exists_in_gazebo(self):
        """检查 Gazebo 中是否存在 mini_diff_robot 模型"""
        try:
            result = subprocess.run(
                ['gz', 'model', '--list'],
                capture_output=True, text=True, timeout=3.0
            )
            return 'mini_diff_robot' in result.stdout
        except Exception:
            return True

    def _cleanup_duplicate_robots(self):
        """移除 Gazebo 中所有 mini_diff_robot 实例"""
        cleanup_count = 0
        for _ in range(30):
            if not self._robot_exists_in_gazebo():
                if cleanup_count > 0:
                    self.node.get_logger().info(f"Cleaned up {cleanup_count} duplicate robot(s)")
                return
            cleanup_count += 1
            self.node.get_logger().warn(f"Removing stale robot instance #{cleanup_count}")
            try:
                subprocess.run(
                    ['gz', 'service', '-s', '/world/default/remove',
                     '--reqtype', 'gz.msgs.Entity',
                     '--reptype', 'gz.msgs.Boolean',
                     '--req', 'name: "mini_diff_robot"\ntype: MODEL',
                     '--timeout', '5000'],
                    timeout=10.0, capture_output=True, text=True
                )
                time.sleep(0.5)
            except Exception:
                break
        if self._robot_exists_in_gazebo():
            self.node.get_logger().error("FAILED to delete robot model after 30 attempts!")

    def _spawn_robot_with_retry(self):
        """生成机器人，含重试机制"""
        for attempt in range(3):
            try:
                result = subprocess.run(
                    self.spawn_robot_cmd,
                    timeout=30.0, capture_output=True, text=True
                )
                if result.returncode == 0:
                    time.sleep(1.0)
                    if self._robot_exists_in_gazebo():
                        self.node.get_logger().info(f"Robot spawned OK (attempt {attempt + 1})")
                        return True
                else:
                    self.node.get_logger().warn(
                        f"Spawn failed (attempt {attempt + 1}): {result.stderr.strip()[:200]}")
            except subprocess.TimeoutExpired:
                self.node.get_logger().warn(f"Spawn TIMED OUT (attempt {attempt + 1})")
            except Exception as e:
                self.node.get_logger().warn(f"Spawn error: {e} (attempt {attempt + 1})")
            if attempt < 2:
                time.sleep(2.0)
        self.node.get_logger().error("FAILED to spawn robot after 3 attempts!")
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. 回合计数 + 初始化
        self._episode_count += 1
        print(f"\n{'='*40}\n=== Episode {self._episode_count} ===\n{'='*40}")
        self.step_count = 0
        self._is_terminated = False
        self.reward = 0.0
        self.goal_dist = self.goal
        self.new_time = 0.0
        self.zhuang = False
        self.min_distance = 10.0
        self._last_collision_time = time.time() - 20.0

        # 2. 停止机器人
        stop_twist = Twist()
        self.vel_pub.publish(stop_twist)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        # 3. 清理 + 重生机器人
        self._cleanup_duplicate_robots()
        time.sleep(1.0)
        if not self._spawn_robot_with_retry():
            self.node.get_logger().error("Reset: spawn FAILED, continuing...")

        # 4. 强制复位到原点
        self.reset_robot_pose(x=0.0, y=0.0, yaw=0.0)

        # 5. 通知 odom_tf_broadcaster
        self.reset_signal_pub.publish(Bool(data=True))

        # 6. 再次停车
        stop_twist = Twist()
        self.vel_pub.publish(stop_twist)

        # 7. 封锁轨迹发布，清空路径
        self._resetting = True
        with self.path_lock:
            self.robot_path = []

        # 8. 等传感器就绪（不发轨迹）
        self._odom_offset_set = False
        for _ in range(30):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if np.any(self.laser_data > 0) and self._odom_offset_set:
                break
        self._resetting = False

        # 9. 确保旧 marker 过期（lifetime=1s，spin 期间已过 ~3s）
        time.sleep(0.5)

        # 10. 初始化位姿和路径
        self.robot_pose = self.initial_pose.copy()
        print(self.robot_pose['x'])
        print(self.robot_pose['y'])
        with self.path_lock:
            self.robot_path = [(self.robot_pose['x'], self.robot_pose['y'])]

        # 11. 设置目标
        x = random.uniform(4, 5)
        y = random.uniform(1, 2)
        self.set_goal(x, y)

        # 12. 角度
        dx = self.current_goal[0] - self.robot_pose['x']
        dy = self.current_goal[1] - self.robot_pose['y']
        self.angle_to_goal = (math.atan2(dy, dx) - self.robot_pose['yaw'] + math.pi) % (2 * math.pi) - math.pi
        self.angle = self.angle_to_goal

        return self._get_obs(), {}

    def _get_obs(self):
        """平坦观测：360 激光 + 3 位姿 = 363 维"""
        laser_data = self.laser_data.astype(np.float32)
        pose_data = np.array([
            self.robot_pose['x'],
            self.robot_pose['y'],
            self.robot_pose['yaw']
        ], dtype=np.float32) if self.robot_pose else np.zeros(3, dtype=np.float32)
        return np.concatenate([laser_data, pose_data])

    def laser_callback(self, msg):
        self.laser_data = np.array(msg.ranges)
        self.laser_data[np.isinf(self.laser_data)] = 10.0
        self.laser_data[self.laser_data < 0.1] = 0.1
        self.laser_data[np.isnan(self.laser_data)] = 10.0

    def step(self, action):
        if self._is_terminated:
            return self._get_obs(), 0, True, False, {}

        twist = Twist()
        if action == 0:  # 慢速前进
            twist.linear.x = 0.4
            twist.angular.z = 0.0
        elif action == 1:  # 快速前进
            twist.linear.x = 0.6
            twist.angular.z = 0.0
        elif action == 2:  # 左转
            twist.linear.x = 0.0
            twist.angular.z = 0.6
        elif action == 3:  # 右转
            twist.linear.x = 0.0
            twist.angular.z = -0.6

        self.vel_pub.publish(twist)
        global TIME_USE
        TIME_USE = time.time()
        new_time = time.time()
        self.step_count += 1
        stop_twist = Twist()

        while time.time() - new_time < 0.05:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        self.vel_pub.publish(stop_twist)
        self.new_time = time.time()
        rclpy.spin_once(self.node, timeout_sec=0.01)

        valid_ranges = self.laser_data[self.laser_data > 0.12 * 1.6]
        min_distance = min(valid_ranges) if len(valid_ranges) > 0 else 10.0

        goal_dist = math.hypot(
            self.current_goal[0] - self.robot_pose['x'],
            self.current_goal[1] - self.robot_pose['y']
        )

        angle_to_goal = math.atan2(
            self.current_goal[1] - self.robot_pose['y'],
            self.current_goal[0] - self.robot_pose['x']
        ) - self.robot_pose['yaw']
        angle_to_goal = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi

        # 奖励函数
        reward = 0.0
        current_time = time.time()

        # 步骤惩罚
        time_penalty = 0.1
        reward -= time_penalty

        # 距离奖励
        distance_reward = 7.0 * (self.goal_dist - goal_dist)
        reward += distance_reward

        # 避障
        obstacle_penalty = 0
        if min_distance > 0.5:
            reward += obstacle_penalty
        else:
            obstacle_penalty = 1.5 * (1.0 - min_distance / 0.5)
            reward -= obstacle_penalty

        # 碰撞
        if min_distance < 0.25 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                self.zhuang = True
                self._last_collision_time = current_time
                global wall
                wall += 1
                reward = -40 - goal_dist * 3
                self.reward += reward
                self._is_terminated = True
                self._record_episode("collision", goal_dist)
                print('碰撞 ' + str(self.step_count) + ' ' + str(self.reward) + ' ' +
                      str(self.robot_pose['x']) + ' ' + str(self.robot_pose['y']) + ' ' +
                      str(self.current_goal[0]) + ' ' + str(self.current_goal[1]) + ' ' + str(goal_dist))
                self.log_event("collision", goal_dist)
            terminated = True

        # 到达
        elif goal_dist < 0.25 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                self._last_collision_time = current_time
                global arrive
                arrive += 1
                reward += 50 + max(0, 150 - self.step_count) * 0.2
                self.reward += reward
                self._is_terminated = True
                self._record_episode("arrive", goal_dist)
                print('到达 ' + str(self.step_count) + ' ' + str(self.reward) + ' ' +
                      str(self.robot_pose['x']) + ' ' + str(self.robot_pose['y']) + ' ' +
                      str(self.current_goal[0]) + ' ' + str(self.current_goal[1]) + ' ' + str(goal_dist))
                self.log_event("arrive", goal_dist)
            terminated = True

        # 超时
        elif self.step_count >= 150 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                self._last_collision_time = current_time
                global timeout
                timeout += 1
                reward += -25 - 2 * goal_dist
                self.reward += reward
                self._is_terminated = True
                self._record_episode("timeout", goal_dist)
                print('超时 ' + str(self.step_count) + ' ' + str(self.reward) + ' ' +
                      str(self.robot_pose['x']) + ' ' + str(self.robot_pose['y']) + ' ' +
                      str(self.current_goal[0]) + ' ' + str(self.current_goal[1]) + ' ' + str(goal_dist))
                self.log_event("timeout", goal_dist)
            terminated = True
        else:
            terminated = False

        self.reward += reward

        if terminated:
            stop_twist = Twist()
            self.vel_pub.publish(stop_twist)
            time.sleep(2.0)

        self.goal_dist = goal_dist
        self.angle_to_goal = angle_to_goal
        self.min_distance = min_distance

        return self._get_obs(), reward, terminated, False, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def odom_callback(self, msg):
        if not self._odom_offset_set:
            self._odom_offset_set = True

        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': self._quaternion_to_yaw(msg.pose.pose.orientation)
        }

        # reset 期间不记录路径、不发布可视化
        if not self._resetting:
            with self.path_lock:
                self.robot_path.append((self.robot_pose['x'], self.robot_pose['y']))
                if len(self.robot_path) > 5000:
                    self.robot_path.pop(0)
            self._publish_visualization()

    def _quaternion_to_yaw(self, quat):
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _record_episode(self, event_type, goal_dist):
        """记录每回合统计"""
        self._episode_stats.append({
            'event': event_type,
            'reward': round(self.reward, 2),
            'steps': self.step_count,
            'goal_dist': round(goal_dist, 2)
        })

    def _publish_visualization(self):
        if not self.robot_pose:
            return

        # 1. 机器人
        marker_array = MarkerArray()
        robot_marker = Marker()
        robot_marker.header.frame_id = "odom"
        robot_marker.header.stamp = self.node.get_clock().now().to_msg()
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CUBE
        robot_marker.pose.position.x = self.robot_pose['x']
        robot_marker.pose.position.y = self.robot_pose['y']
        robot_marker.pose.position.z = 0.1
        robot_marker.pose.orientation.z = math.sin(self.robot_pose['yaw'] / 2)
        robot_marker.pose.orientation.w = math.cos(self.robot_pose['yaw'] / 2)
        robot_marker.scale.x = 0.3
        robot_marker.scale.y = 0.2
        robot_marker.scale.z = 0.1
        robot_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        marker_array.markers.append(robot_marker)
        self.robot_markers_pub.publish(marker_array)

        # 2. 轨迹（带 lifetime，reset 期间自动过期）
        if len(self.robot_path) > 1:
            path_marker = Marker()
            path_marker.header.frame_id = "odom"
            path_marker.header.stamp = self.node.get_clock().now().to_msg()
            path_marker.ns = "trajectory"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.scale.x = 0.02
            path_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
            path_marker.lifetime = Duration(sec=1, nanosec=0)
            for x, y in self.robot_path:
                path_marker.points.append(Point(x=x, y=y, z=0.05))
            self.trajectory_pub.publish(path_marker)

        # 3. 目标点（带 lifetime）
        if self.current_goal:
            goal_marker = Marker()
            goal_marker.header.frame_id = "odom"
            goal_marker.header.stamp = self.node.get_clock().now().to_msg()
            goal_marker.ns = "goal"
            goal_marker.id = 0
            goal_marker.type = Marker.SPHERE
            goal_marker.pose.position.x = self.current_goal[0]
            goal_marker.pose.position.y = self.current_goal[1]
            goal_marker.pose.position.z = 0.1
            goal_marker.scale.x = 0.45
            goal_marker.scale.y = 0.45
            goal_marker.scale.z = 0.3
            goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            goal_marker.lifetime = Duration(sec=1, nanosec=0)
            self.goal_pub.publish(goal_marker)

    def log_event(self, event_type, goal_dist):
        msg = f"{event_type} | step: {self.step_count} | reward: {self.reward:.2f} | " \
              f"robot: ({self.robot_pose['x']:.2f},{self.robot_pose['y']:.2f}) | " \
              f"goal: ({self.current_goal[0]:.2f},{self.current_goal[1]:.2f}) | dist: {goal_dist:.2f}"
        print(msg)
        with open(self.event_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event_type, self.step_count, self.reward,
                self.robot_pose["x"], self.robot_pose["y"],
                self.current_goal[0], self.current_goal[1], goal_dist
            ])


class CmdVelGuard1():
    def __init__(self):
        super().__init__()

    def printall(self):
        global arrive
        print(arrive)
        global wall
        print(wall)
        global timeout
        print(timeout)
