import random

from stable_baselines3 import PPO
from fishbot_application.robot_env import RobotEnv, CmdVelGuard
from fishbot_application.robot_env1 import RobotEnv1, CmdVelGuard1
from fishbot_application.train import LSTMPolicy,LSTMFeatureExtractor,CustomMLPExtractor
from fishbot_application.train import BiLSTM_Timing_Policy
import gymnasium as gym
import rclpy
import time

def main():

    # 初始化ROS 2
    rclpy.init()
    guard = CmdVelGuard1()
    
    try:
        # 创建环境并设置目标
        env = RobotEnv1()
        # env.set_goal(x=5.0,y=1.0)
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        env.set_goal(x=x1,y=y1)
        # env = gym.wrappers.Autoreset(env)
        
        # 加载模型
        custom_objects = {
            "LSTMPolicy": LSTMPolicy,
            "LSTMFeatureExtractor": LSTMFeatureExtractor,
            "CustomMLPExtractor": CustomMLPExtractor
        }
        # model=PPO.load("move1_robot230",env=env,custom_objects=custom_objects,policy=LSTMPolicy)
        # model=PPO.load("BiLSTM_robot400",env=env,policy=BiLSTM_Timing_Policy)
        model=PPO.load("PPO_robot510",env=env)
        model.policy.set_training_mode(False)  # 评估模式

        # model.policy.reset_hidden_state(batch_size=1)
        env.model = model
        obs,_= env.reset()
        for step in range(20000):
            action, _ = model.predict(obs,deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, _ = env.reset()
                # model.policy.reset_hidden_state(batch_size=1)
        guard.printall()

    finally:
        env.close()

if __name__ == '__main__':
    main()