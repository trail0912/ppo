import random
import time
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, PoseStamped,Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool
import math
from threading import Lock
from std_srvs.srv import Empty
import threading
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from ros_gz_interfaces.msg import WorldControl, Entity
import subprocess

import os
import csv
from datetime import datetime

TIME_USE=0.0
arrive=0
wall=0
timeout=0

class RobotEnv(gym.Env):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        super(RobotEnv, self).__init__()
        self.node = rclpy.create_node('robot_env')

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

        self.world_control_client = self.node.create_client(
            ControlWorld,
            '/world/default/control'
        )

        while not self.world_control_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                '/world/default/control not available, waiting...'
            )

        # 新增：用于每轮 reset 时强制设置机器人位置
        self.set_pose_client = self.node.create_client(
            SetEntityPose,
            '/world/default/set_pose'
        )
        while not self.set_pose_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                '/world/default/set_pose not available, waiting...'
            )

        # 确保初始位姿有效
        self.initial_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.robot_pose = self.initial_pose.copy()

        # odom 偏移：新机器人 odom 不一定从 0 开始，用差值得到相对位移
        self._odom_offset_x = 0.0
        self._odom_offset_y = 0.0
        self._odom_offset_set = False

        # 数据存储
        self.laser_data = np.zeros(360)
        self.robot_path = []
        self.current_goal = None
        self.path_lock = Lock()
        self.step_count = 0
        self._is_terminated = False
        self._last_collision_time = 0
        self.goal=0
        self.goal_dist=0
        self.reward=0
        self.angle_to_goal=0
        self.angle=0
        self.zhuang=False
        self.min_distance=0
        self.obs_history = []  # 存储历史观测
        self.seq_len = 11  # 序列长度（取最近4步观测，可调整）
        self.obs_dim = 365
        self.new_time=0
        self.distance_reward=0
        self.obstacle_penalty=0
        self.spawn_robot_cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-topic', '/robot_description',
            '-name', 'mini_diff_robot'
        ]

        # 动作和观测空间
        self.action_space = gym.spaces.Discrete(4)  # 0:前进, 1:左转, 2:右转
        # 在__init__中修改观测空间，匹配时序输入形状
        # self.observation_space = gym.spaces.Box(
        #     low=np.tile(np.array([0.0]*360 + [-10.0, -10.0, -np.pi], dtype=np.float32), (self.seq_len, 1)),  # 重复seq_len次
        #     high=np.tile(np.array([10.0]*360 + [10.0, 10.0, np.pi], dtype=np.float32), (self.seq_len, 1)),
        #     shape=(self.seq_len, self.obs_dim),  # 时序形状：(seq_len, 363)
        #     dtype=np.float32
        # )
        self.observation_space = gym.spaces.Box(
        # 修正：360(激光)+3(位姿)+2(目标) = 365维
            low=np.tile(np.array([0.0]*360 + [-10.0, -10.0, -np.pi, 0.0, -np.pi], dtype=np.float32), (self.seq_len, 1)),
            high=np.tile(np.array([10.0]*360 + [10.0, 10.0, np.pi, 10.0, np.pi], dtype=np.float32), (self.seq_len, 1)),
            shape=(self.seq_len, self.obs_dim),  # 现在low/high和shape维度完全匹配
            dtype=np.float32
        )


        # 初始化日志文件（放在 __init__）
        self.log_dir = "train_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_log_path = os.path.join(self.log_dir, f"navigation_events_{time_str}.csv")

        # 写表头
        with open(self.event_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_type", "step_count", "episode_reward",
                "robot_x", "robot_y", "goal_x", "goal_y", "goal_dist"
            ])



    def set_goal(self, x, y):
        """设置训练目标点"""
        self.current_goal = (x, y)
        self.goal = math.hypot(
            x - self.robot_pose['x'],
            y - self.robot_pose['y']
        )
        self.goal_dist=self.goal
        self._publish_visualization()

    #added
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

    def _delete_robot_model(self):
        """删除 Gazebo 中旧的 mini_diff_robot 模型，防止累积"""
        try:
            result = subprocess.run(
                ['gz', 'service', '-s', '/world/default/remove',
                 '--reqtype', 'gz.msgs.Entity',
                 '--reptype', 'gz.msgs.Boolean',
                 '--req', 'name: "mini_diff_robot"\ntype: MODEL',
                 '--timeout', '5000'],
                timeout=10.0, capture_output=True, text=True
            )
            if result.returncode == 0:
                self.node.get_logger().info("Old robot model deleted OK")
            else:
                self.node.get_logger().warn(
                    f"gz service remove failed: rc={result.returncode} "
                    f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn("gz service remove TIMED OUT")
        except FileNotFoundError:
            self.node.get_logger().warn("gz command NOT FOUND")
        except Exception as e:
            self.node.get_logger().warn(f"gz service remove error: {e}")

    def reset(self, seed=None, options=None):
        """重置环境，不依赖 Gazebo 服务"""
        super().reset(seed=seed)

        # 1. 初始化步骤计数和状态
        self.step_count = 0
        self._is_terminated = False
        self.reward = 0.0
        self.goal_dist = self.goal
        self.obs_history = []
        self.new_time = 0.0
        self.zhuang = False
        self.min_distance = 10.0

        self._last_collision_time = time.time() - 20.0 #added

        # 2. 停止机器人
        stop_twist = Twist()
        self.vel_pub.publish(stop_twist)
        rclpy.spin_once(self.node, timeout_sec=0.1)

        # 3. 删除旧机器人 + 生成新机器人（不重置世界，避免 sim time 跳变）
        self._delete_robot_model()
        subprocess.run(self.spawn_robot_cmd, timeout=30.0)

        # 4. 强制把机器人放回原点
        self.reset_robot_pose(x=0.0, y=0.0, yaw=0.0)

        # 5. 通知 odom_tf_broadcaster：robot 已 reset
        self.reset_signal_pub.publish(Bool(data=True))

        # 6. reset 后再次停车
        stop_twist = Twist()
        self.vel_pub.publish(stop_twist)

        # 7. 清除旧轨迹 marker
        self._clear_trajectory_marker()

        # 8. 等待传感器 + 里程计就绪，捕获 odom 偏移
        self._odom_offset_set = False
        for _ in range(30):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if np.any(self.laser_data > 0) and self._odom_offset_set:
                break

        # 9. 重置机器人位姿（基于 odom 偏移后的值）
        self.robot_pose = self.initial_pose.copy()

        print(self.robot_pose['x'])
        print(self.robot_pose['y'])

        with self.path_lock:
            self.robot_path = [
                (self.robot_pose['x'],
                self.robot_pose['y'])
            ]

        # 5. 设置新目标点
        # x = random.uniform(-5, 5)
        # y = random.uniform(-5, 5)
        # while math.hypot(x, y) < 1.0:
        #     x = random.uniform(-5, 5)
        #     y = random.uniform(-5, 5)
        x = random.uniform(4, 5)
        y = random.uniform(1, 2)
        self.set_goal(x, y)

        # 6. 更新角度相关信息
        dx = self.current_goal[0] - self.robot_pose['x']
        dy = self.current_goal[1] - self.robot_pose['y']
        self.angle_to_goal = (math.atan2(dy, dx) - self.robot_pose['yaw'] + math.pi) % (2*math.pi) - math.pi
        self.angle = self.angle_to_goal

        # 8. 重置时序观测队列
        obs = self._get_obs()

        # 9. 如果使用模型隐藏状态（RNN/LSTM），重置隐藏状态
        if hasattr(self, 'model') and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'reset_hidden_state'):
            self.model.policy.reset_hidden_state(batch_size=1)

        return obs, {}

    def _check_ready(self):
        """检查是否完成重置"""
        return (
            self.robot_pose is not None
            and np.any(self.laser_data > 0)  # 激光数据已更新
        )


    def _get_obs(self):
        """将激光、位姿、目标点相对信息拼接为时序观测（适配随机目标点）"""
        # 1. 原有激光数据（假设laser_data是360维激光雷达数据）
        laser_data = self.laser_data.astype(np.float32)  # 360维

        # 2. 原有位姿数据（x,y,yaw → 3维）
        pose_data = np.array([
            self.robot_pose['x'],
            self.robot_pose['y'],
            self.robot_pose['yaw']
        ], dtype=np.float32) if self.robot_pose else np.zeros(3, dtype=np.float32)  # 3维

        # 3. 新增：当前随机目标点的相对观测（核心！适配随机目标点）
        if hasattr(self, 'current_goal') and self.current_goal is not None:
            # 计算机器人到当前目标点的相对距离和角度
            dx = self.current_goal[0] - self.robot_pose['x']
            dy = self.current_goal[1] - self.robot_pose['y']
            # 相对距离（标量）
            goal_dist = np.hypot(dx, dy).astype(np.float32)
            # 相对角度（机器人朝向与目标点的夹角，归一化到[-π, π]）
            goal_angle = np.arctan2(dy, dx) - self.robot_pose['yaw']
            goal_angle = np.clip(goal_angle, -np.pi, np.pi).astype(np.float32)
            # 目标点观测（2维）
            goal_data = np.array([goal_dist, goal_angle], dtype=np.float32)
        else:
            # 无目标点时填充0（仅初始化阶段）
            goal_data = np.zeros(2, dtype=np.float32)

        # 4. 拼接单步观测：激光(360) + 位姿(3) + 目标点(2) = 365维
        current_obs = np.concatenate([laser_data, pose_data, goal_data])

        # 5. 维护时序观测队列（原有逻辑不变）
        self.obs_history.append(current_obs)
        if len(self.obs_history) > self.seq_len:
            self.obs_history.pop(0)  # 只保留最近seq_len步

        # 若序列长度不足，用初始观测（0向量）填充
        while len(self.obs_history) < self.seq_len:
            self.obs_history.insert(0, np.zeros_like(current_obs))

        # 6. 返回形状为 (seq_len, 365) 的时序观测（原363→365，适配新增的2维目标信息）
        return np.array(self.obs_history, dtype=np.float32)

    def laser_callback(self, msg):
        # 处理激光数据
        self.laser_data = np.array(msg.ranges)
        # 1. 无穷大 → 最大有效距离（如 10.0）
        self.laser_data[np.isinf(self.laser_data)] = 10.0
        # 2. 小于 0.1 的值 → 0.1（避免过近的噪声，如传感器盲区）
        self.laser_data[self.laser_data < 0.1] = 0.1
        # 3. NaN → 10.0（极少数情况，保险处理）
        self.laser_data[np.isnan(self.laser_data)] = 10.0

    def step(self, action):
        # 执行动作
        if self._is_terminated:  # 如果已终止，直接返回
            return self._get_obs(), 0, True, False, {}

        twist = Twist()
        if action == 0:  # 慢速前进
            twist.linear.x = 0.4
            twist.angular.z = 0.0
            # print(0)
        elif action == 1:  # 快速前进
            twist.linear.x = 0.6
            twist.angular.z = 0.0
            # print(1)
        elif action == 2:  # 小半径左转（保留原设计）
            twist.linear.x = 0.0
            twist.angular.z = 0.6
            # print(4)
        elif action == 3:  # 小半径右转（保留原设计）
            twist.linear.x = 0.0
            twist.angular.z = -0.6
            # print(5)

        # print(self.robot_pose['x'])
        # print(self.robot_pose['y'])


        # while time.time()-self.new_time <0.15:
        #     rclpy.spin_once(self.node, timeout_sec=0.01)

        self.vel_pub.publish(twist)
        global TIME_USE
        TIME_USE=time.time()
        new_time=time.time()
        self.step_count += 1
        stop_twist = Twist()

        while time.time()-new_time <0.05:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        self.vel_pub.publish(stop_twist)

        self.new_time=time.time()

        rclpy.spin_once(self.node, timeout_sec=0.01)

        valid_ranges = self.laser_data[self.laser_data > 0.12 * 1.6]  # 忽略接近 range_min 的值
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

        # 复合奖励函数
        reward = 0

        obstacle_penalty=0

        current_time = time.time()

            # 基础奖励初始化
        reward = 0.0

        # 1. 步骤惩罚（鼓励高效）
        time_penalty=0.1
        reward -= time_penalty

        distance_reward=7.0 * (self.goal_dist - goal_dist)
        reward += distance_reward  # 靠近目标得正奖，远离得负奖

        # 3. 避障辅助奖励（安全距离）
        if min_distance > 0.5:
            # obstacle_penalty=0.04
            reward += obstacle_penalty  # 安全距离奖励
        else:
            obstacle_penalty = 1.5 * (1.0 - min_distance / 0.5)
            reward-=obstacle_penalty

        # print('------------------------------------------------')
        # print(time_penalty)
        # print(distance_reward)
        # print(obstacle_penalty)
        # print(reward)

        # print(5.0 * (self.goal_dist - goal_dist))

        # self.distance_reward+=distance_reward
        # self.obstacle_penalty+=obstacle_penalty

        # 碰撞惩罚
        if min_distance < 0.25 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                    # print(self.step_count)
                    self.zhuang=True
                    self._last_collision_time = current_time
                    global wall
                    wall+=1
                    reward =-40-goal_dist*3
                    self.reward+=reward
                    self._is_terminated = True
                    #print('碰撞 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
                    self.log_event("collision", goal_dist)
            terminated = True

        # 到达目标奖励
        elif goal_dist < 0.25 and current_time - self._last_collision_time > 10.0:  # 到达阈值
            if not self._is_terminated:
                # print(self.step_count)
                self._last_collision_time = current_time
                global arrive
                arrive+=1
                reward += 50+ max(0, 150 - self.step_count) * 0.2
                self.reward+=reward
                self._is_terminated = True
                #print('到达 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
                self.log_event("arrive", goal_dist)
            terminated = True
        elif self.step_count >= 150 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                self._last_collision_time = current_time
                global timeout
                timeout+=1
                reward += -25-2*goal_dist
                self.reward+=reward
                self._is_terminated = True
                #print('超时 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
                self.log_event("timeout", goal_dist)
            terminated = True
        else:
            terminated=False
            # print('------------------------------------------------')
            # print(action)
            # print(distance_reward)
            # print(heading_reward)
            # print(time_penalty)
            # print(movement_reward)
            # print(min_distance)
            # print(obstacle_penalty)
            # print(goal_reward)
            # print(reward)
            # print(self.reward)

        self.reward+=reward

        # 如果碰撞，立即停止机器人
        if terminated:
            stop_twist = Twist()
            self.vel_pub.publish(stop_twist)
            time.sleep(2.0)

        self.goal_dist=goal_dist
        self.angle_to_goal=angle_to_goal
        self.min_distance=min_distance

        # print(distance_reward)
        # print(heading_reward)

        return self._get_obs(), reward, terminated, False, {}

    def close(self):
        # 清理资源
        self.node.destroy_node()
        rclpy.shutdown()

    def odom_callback(self, msg):
        # 捕获 odom 偏移（第一个 odom 消息的值作为本回合零点）
        if not self._odom_offset_set:
            self._odom_offset_x = msg.pose.pose.position.x
            self._odom_offset_y = msg.pose.pose.position.y
            self._odom_offset_set = True

        # 更新机器人位姿（使用 odom 偏移后的相对位置）
        self.robot_pose = {
            'x': msg.pose.pose.position.x - self._odom_offset_x,
            'y': msg.pose.pose.position.y - self._odom_offset_y,
            'yaw': self._quaternion_to_yaw(msg.pose.pose.orientation)
        }

        # TF 由 launch 的 odom_tf_broadcaster 统一发布

        # 记录路径
        with self.path_lock:
            self.robot_path.append((self.robot_pose['x'], self.robot_pose['y']))
            if len(self.robot_path) > 5000:  # 限制路径长度
                self.robot_path.pop(0)

        # 更新可视化
        self._publish_visualization()

    def _quaternion_to_yaw(self, quat):
        # 四元数转偏航角
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def _clear_trajectory_marker(self):
        """清除 RViz 中旧的轨迹 marker"""
        clear = Marker()
        clear.header.frame_id = "odom"
        clear.ns = "trajectory"
        clear.id = 0
        clear.action = Marker.DELETE
        self.trajectory_pub.publish(clear)

    def _publish_visualization(self):
        if not self.robot_pose:
            return

        g = self.current_goal
        self.node.get_logger().info(
            f'VIZ: goal=({g[0]:.1f},{g[1]:.1f})' if g else 'VIZ: no goal',
            throttle_duration_sec=3.0)

        # 1. 机器人当前状态标记
        marker_array = MarkerArray()

        # 机器人本体（蓝色立方体）
        robot_marker = Marker()
        robot_marker.header.frame_id = "odom"
        now = self.node.get_clock().now().to_msg()
        robot_marker.header.stamp = now
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CUBE
        robot_marker.pose.position.x = self.robot_pose['x']
        robot_marker.pose.position.y = self.robot_pose['y']
        robot_marker.pose.position.z = 0.1
        robot_marker.pose.orientation.z = math.sin(self.robot_pose['yaw']/2)
        robot_marker.pose.orientation.w = math.cos(self.robot_pose['yaw']/2)
        robot_marker.scale.x = 0.3
        robot_marker.scale.y = 0.2
        robot_marker.scale.z = 0.1
        robot_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        marker_array.markers.append(robot_marker)

        self.robot_markers_pub.publish(marker_array)

        # 2. 机器人轨迹（绿色线条）
        if len(self.robot_path) > 1:
            path_marker = Marker()
            path_marker.header.frame_id = "odom"
            now = self.node.get_clock().now().to_msg()
            path_marker.header.stamp = now
            path_marker.ns = "trajectory"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.scale.x = 0.02
            path_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)

            for x, y in self.robot_path:
                path_marker.points.append(Point(x=x, y=y, z=0.05))

            self.trajectory_pub.publish(path_marker)

        # 3. 目标点（红色球体）
        if self.current_goal:
            goal_marker = Marker()
            goal_marker.header.frame_id = "odom"
            now = self.node.get_clock().now().to_msg()
            goal_marker.header.stamp = now
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
            self.goal_pub.publish(goal_marker)

    # 增加一个方法同时打印和写日志
    def log_event(self, event_type, goal_dist):
        msg = f"{event_type} | step: {self.step_count} | reward: {self.reward:.2f} | " \
            f"robot: ({self.robot_pose['x']:.2f},{self.robot_pose['y']:.2f}) | " \
            f"goal: ({self.current_goal[0]:.2f},{self.current_goal[1]:.2f}) | dist: {goal_dist:.2f}"

        # 打印到终端
        print(msg)

        # 写入 CSV
        with open(self.event_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event_type,
                self.step_count,
                self.reward,
                self.robot_pose["x"],
                self.robot_pose["y"],
                self.current_goal[0],
                self.current_goal[1],
                goal_dist
            ])


class CmdVelGuard():
    def __init__(self):
        super().__init__()

    def printall(self):
        global arrive
        print(arrive)
        global wall
        print(wall)
        global timeout
        print(timeout)
