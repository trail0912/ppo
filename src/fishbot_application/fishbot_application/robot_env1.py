import random
import time
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, PoseStamped,Quaternion, TransformStamped 
from gazebo_msgs.srv import SetEntityState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math
from threading import Lock
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from gazebo_msgs.msg import ModelStates
import threading

TIME_USE=0.0
arrive=0
wall=0
timeout=0

class RobotEnv1(gym.Env):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        
        super(RobotEnv1, self).__init__()
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

         # 添加重置服务客户端
        self.reset_sim_client = self.node.create_client(Empty, '/reset_simulation')
        self.reset_world_client = self.node.create_client(Empty, '/reset_world')

        self.tf_broadcaster = TransformBroadcaster(self.node)
        
        # 确保初始位姿有效
        self.initial_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.robot_pose = self.initial_pose.copy()
        
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
        self.new_time=0
        
        # 动作和观测空间
        self.action_space = gym.spaces.Discrete(4)  # 0:前进, 1:左转, 2:右转
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*360 + [-10.0, -10.0, -np.pi], dtype=np.float32),
            high=np.array([10.0]*360 + [10.0, 10.0, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        # self.observation_space = gym.spaces.Box(low=0.0, high=10.0, shape=(360,), dtype=np.float32)

    def set_goal(self, x, y):
        """设置训练目标点"""
        self.current_goal = (x, y)
        self.goal = math.hypot(
            x - self.robot_pose['x'],
            y - self.robot_pose['y']
        )
        self.goal_dist=self.goal
        self._publish_visualization()

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     with self.path_lock:
    #         self.robot_path = []
    #     return self._get_obs(), {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # pairs = [1, 2, 3]
        # a=random.choice(pairs)

        # if a==1:
        #     self.set_goal(x=3.0,y=4.0)
        # elif a==2:
        #     self.set_goal(x=4.0,y=3.0)
        # elif a==3:
        #     self.set_goal(x=4.0,y=4.0)

        # x = random.uniform(-2, 2)
        # y = random.uniform(-2, 2)
        # self.set_goal(x,y)

        self.step_count = 0
        self._is_terminated = False
        self.goal_dist=self.goal
        self.reward=0
        self.zhuang=False
        self.min_distance=10.0
        self.new_time=0

        # 1. 停止机器人
        stop_twist = Twist()
        self.vel_pub.publish(stop_twist)

        time.sleep(2.0)

        self.reset_world_client.call_async(Empty.Request())

        self.vel_pub.publish(stop_twist)

        time.sleep(2.0)
        
        # 3. 强制刷新传感器数据
        self.laser_data = np.zeros(360)  # 清空激光数据
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self._publish_reset_tf()
        
        # 4. 重置内存状态
        self.robot_pose = self.initial_pose.copy()
        with self.path_lock:
            self.robot_path = []

        # x = random.uniform(4, 5)
        # y = random.uniform(1, 2)
        # self.set_goal(x,y)
        self.set_goal(5.0,1.0)   

        self.angle_to_goal= math.atan2(
            self.current_goal[1] - self.robot_pose['y'],
            self.current_goal[0] - self.robot_pose['x']
        ) - self.robot_pose['yaw']
        self.angle_to_goal = (self.angle_to_goal + math.pi) % (2 * math.pi) - math.pi
        self.angle=self.angle_to_goal
        
        # 5. 返回最新观测
        return self._get_obs(), {}
    
    def _publish_reset_tf(self):
        # 发布odom->base_link的零位变换
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_footprint"
        tf_msg.transform.rotation.w = 0.0  # 无旋转
        self.tf_broadcaster.sendTransform(tf_msg)
    
    def _check_ready(self):
        """检查是否完成重置"""
        return (
            self.robot_pose is not None 
            and np.any(self.laser_data > 0)  # 激光数据已更新
        )

    # def _get_obs(self):
    #     """获取当前观测"""
    #     return {
    #         "laser": self.laser_data,
    #         "pose": np.array([
    #             self.robot_pose['x'],
    #             self.robot_pose['y'],
    #             self.robot_pose['yaw']
    #         ]) if self.robot_pose else np.zeros(3)
    #     }

    def _get_obs(self):
        """将激光和位姿拼接为单一数组"""
        laser_data = self.laser_data.astype(np.float32)  # 确保类型匹配
        pose_data = np.array([
            self.robot_pose['x'],
            self.robot_pose['y'],
            self.robot_pose['yaw']
        ], dtype=np.float32) if self.robot_pose else np.zeros(3, dtype=np.float32)
       
        return np.concatenate([laser_data, pose_data])  # 363维数组 (360+3)

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

        while time.time()-self.new_time <0.15:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        self.vel_pub.publish(twist)
        global TIME_USE
        TIME_USE=time.time()
        new_time=time.time()
        self.step_count += 1
        stop_twist = Twist()

        while time.time()-new_time <0.3:
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
         # 1. 距离奖励（改进方案）
        # distance_reward = (self.goal_dist- goal_dist) * 30

        # print(distance_reward)

        # print(goal_dist)
    
    # 2. 航向奖励（连续平滑）
        # if abs(angle_to_goal) < math.pi/6:
        #     heading_reward = 1.57 + 5 * (math.pi/6 - abs(angle_to_goal))
        # elif abs(angle_to_goal) < math.pi/2:   # <30°: 良好对准
        #     heading_reward = 1.5 * (math.pi/2 - abs(angle_to_goal))
        # else:   # <30°: 良好对准
        #     heading_reward = -2.5 * (abs(angle_to_goal) - math.pi/2)

        # heading_reward/=1.5

        # print(heading_reward)

        # print(action)
        # print(self.angle_to_goal)
        # print(angle_to_goal)

        # time_penalty=-0.5

        # movement_reward=0
        # if action == 0 or action == 1:  # 只有前进时检查方向
        #     if abs(self.angle_to_goal) < math.pi/36:
        #         movement_reward = 5.0
        #     elif abs(self.angle_to_goal) < math.pi/18:   # <30°: 良好对准
        #         movement_reward = 2.5
        #     elif abs(self.angle_to_goal) < math.pi/9:   # <30°: 良好对准
        #         movement_reward = 1.0
        #     elif abs(self.angle_to_goal) < math.pi/4:                                  # 其他情况：惩罚
        #         movement_reward = 0
        #     elif abs(self.angle_to_goal) > math.pi/3 and abs(self.angle_to_goal) <= math.pi/2:
        #         movement_reward = -2.0
        #     elif abs(self.angle_to_goal) > math.pi/2 and abs(self.angle_to_goal) <= math.pi/1.5:  
        #         movement_reward = -4.5
        #     elif abs(self.angle_to_goal) > math.pi/1.5:  
        #         movement_reward = -6.0
        #     else:
        #         movement_reward = -1.0
        #     # print(movement_reward)
        # elif action == 2 or action == 3:
        #     if abs(self.angle_to_goal-angle_to_goal)>math.pi/90:
        #         if abs(self.angle_to_goal) < abs(angle_to_goal):
        #             movement_reward=-3.5
        #         elif abs(self.angle_to_goal) >= abs(angle_to_goal):
        #             movement_reward=2.0
            # print(str(action)+ ' ' + str(self.angle_to_goal)+' '+str(movement_reward)+' '+str(angle_to_goal))

        obstacle_penalty=0
        # if min_distance<0.55:
        #     obstacle_penalty= -3 - 30 * (0.55 - min_distance)
        # elif min_distance<0.7:
        #     obstacle_penalty= -20 * (0.7 - min_distance)
        # else:
        #     if min_distance<0.4:
        #         obstacle_penalty= (min_distance-self.min_distance) * 50

        # print(obstacle_penalty)
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

        # reward = (distance_reward + heading_reward + obstacle_penalty + movement_reward + time_penalty)

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
                    print('碰撞 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
            terminated = True
            
        # 到达目标奖励
        elif goal_dist < 0.25 and current_time - self._last_collision_time > 10.0:  # 到达阈值
            if not self._is_terminated:
                # print(self.step_count)
                self._last_collision_time = current_time
                global arrive
                arrive+=1
                reward += 50+ max(0, 150 - self.step_count) * 0.2
                # reward += max(-100,(250-self.step_count)*5)
                self.reward+=reward
                self._is_terminated = True
                print('到达 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
            terminated = True
        elif self.step_count >= 150 and current_time - self._last_collision_time > 10.0:
            if not self._is_terminated:
                self._last_collision_time = current_time
                global timeout
                timeout+=1
                reward += -25-2*goal_dist
                # if reward<-450:
                #     reward=-450
                self.reward+=reward
                self._is_terminated = True
                print('超时 '+str(self.step_count)+' '+str(self.reward)+' '+str(self.robot_pose['x'])+' '+str(self.robot_pose['y'])+' '+str(self.current_goal[0])+' '+ str(self.current_goal[1])+' '+str(goal_dist))
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
        # 更新机器人位姿
        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': self._quaternion_to_yaw(msg.pose.pose.orientation)
        }
        # print(msg.pose.pose.position.x)
        # print(msg.pose.pose.position.y)
            
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

    def _publish_visualization(self):
        if not self.robot_pose:
            return
        
        # 1. 机器人当前状态标记
        marker_array = MarkerArray()
        
        # 机器人本体（蓝色立方体）
        robot_marker = Marker()
        robot_marker.header.frame_id = "odom"
        robot_marker.header.stamp = self.node.get_clock().now().to_msg()
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
            path_marker.header.stamp = self.node.get_clock().now().to_msg()
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
            self.goal_pub.publish(goal_marker)

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
    