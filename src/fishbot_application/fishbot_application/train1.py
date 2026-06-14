import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
import torch
from fishbot_application.robot_env import RobotEnv,CmdVelGuard
# from fishbot_application.robot_env1 import RobotEnv1,CmdVelGuard1
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium as gym
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.duration import Duration
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer  # 基于旧版基础回放池扩展
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomMLPExtractor(nn.Module):
    """
    适配LSTM特征的MLP提取器（完整版）：
    1. 等维回流多尺度特征映射（共享增强层）：64→128→64
    2. 分离策略（actor）和价值（critic）分支，避免梯度冲突
    3. 使用Tanh激活，适配LSTM输出的[-1,1]范围特征
    """
    def __init__(self, features_dim=64, use_backflow=False):
        super().__init__()
        self.use_backflow = use_backflow

        # ===== 等维回流多尺度特征增强（共享层）=====
        if use_backflow:
            self.backflow = nn.Sequential(
                nn.Linear(features_dim, 128),
                nn.Tanh(),
                nn.Linear(128, features_dim),   # 回降至原始维度，形成闭环
                nn.Tanh()
            )
        else:
            self.backflow = nn.Identity()

        # 策略分支（独立）
        self.pi = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        # 价值分支（独立）
        self.vf = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

    def forward(self, features):
        """先经过等维回流增强，再分别送入策略和价值分支"""
        enhanced = self.backflow(features)   # 共享增强
        return self.pi(enhanced), self.vf(enhanced)

    def forward_actor(self, features):
        enhanced = self.backflow(features)
        return self.pi(enhanced)

    def forward_critic(self, features):
        enhanced = self.backflow(features)
        return self.vf(enhanced)

class LSTMPolicy(ActorCriticPolicy):
    """
    适配LSTM的PPO Actor-Critic策略：
    1. 维护每个episode/批次的独立隐藏状态（避免全局共享）
    2. 显式管理隐藏状态的初始化、更新、重置流程
    3. 兼容Stable Baselines3的PPO训练框架
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            # 指定LSTM特征提取器
            features_extractor_class=LSTMFeatureExtractor,
            # 传递LSTM参数（特征维度=64，与MLP提取器匹配）
            features_extractor_kwargs=dict(features_dim=64),
            # 其他PPO默认参数（可根据需求调整）
            net_arch=None,  # 禁用默认MLP，使用自定义CustomMLPExtractor
            activation_fn=nn.Tanh,
            *args, **kwargs
        )
        # 策略级维护当前隐藏状态（单环境=1，多环境=环境数量）
        # 初始为None，调用reset_hidden_state后初始化
        self.current_hidden = None
        self.current_cell = None

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPExtractor(features_dim=self.features_dim, use_backflow=False)

    def extract_features(self, obs):
        """
        明确调用 return_hidden=True，获取隐藏状态并更新，仅返回特征给SB3
        """
        batch_size = obs.shape[0]
        # 1. 若current_hidden未初始化/维度错误，重新初始化
        if (self.current_hidden is None) or (self.current_hidden.shape[1] != batch_size):
            self.reset_hidden_state(batch_size=batch_size)
        # 调用LSTMFeatureExtractor，获取3个返回值
        features, updated_hidden, updated_cell = self.features_extractor(
            obs, self.current_hidden, self.current_cell, return_hidden=True  # 关键：开启return_hidden
        )
        # # 更新策略内部的隐藏状态（仅用于策略预测）
        # self.current_hidden = updated_hidden
        # self.current_cell = updated_cell
        self.current_hidden = updated_hidden.detach()
        self.current_cell = updated_cell.detach()
        return features  # 仅返回特征张量，符合SB3对MLP提取器的输入要求

    def reset_hidden_state(self, batch_size=1):
        """
        重置隐藏状态
        """
        # self.current_hidden, self.current_cell = self.features_extractor.init_hidden_state(
        # batch_size=batch_size
        # )
        # # 新增：detach()确保初始状态无梯度
        # self.current_hidden = self.current_hidden.detach()
        # self.current_cell = self.current_cell.detach()
        self.current_hidden, self.current_cell = self.features_extractor.init_hidden_state(
        batch_size=batch_size
        )
        # 新增：detach()确保初始状态无梯度 + 设备对齐
        self.current_hidden = self.current_hidden.detach()
        self.current_cell = self.current_cell.detach()
        # 核心修复：移到策略的设备（CPU/GPU）
        device = next(self.parameters()).device
        self.current_hidden = self.current_hidden.to(device)
        self.current_cell = self.current_cell.to(device)

    def _predict(self, observation, deterministic=False):
        # 1. 若隐藏状态未初始化，自动初始化（保持原逻辑）
        if self.current_hidden is None:
            batch_size = observation.shape[0]
            self.reset_hidden_state(batch_size=batch_size)

        # 2. 调用extract_features（仅传obs，内部自动处理隐藏状态）
        lstm_features = self.extract_features(observation)

        # 3. 后续逻辑不变（获取策略特征、生成动作）
        pi_features = self.mlp_extractor.forward_actor(lstm_features)
        action_dist = self._get_action_dist_from_latent(pi_features)
        actions = action_dist.get_actions(deterministic=deterministic)

        return actions

    def evaluate_actions(self, obs, actions):
        # batch_size = obs.shape[0]
        # # 1. 为当前批次独立初始化隐藏状态（原逻辑不变）
        # batch_hidden, batch_cell = self.features_extractor.init_hidden_state(batch_size=batch_size)

        # # 2. 调用LSTMFeatureExtractor，开启return_hidden=True
        # lstm_features, _, _ = self.features_extractor(
        #     obs, batch_hidden, batch_cell, return_hidden=True  # 关键：开启return_hidden
        # )
        batch_size = obs.shape[0]
        # 修复：不再重新初始化，而是复用和推理一致的隐藏状态逻辑
        # 步骤1：获取当前批次的初始隐藏状态（若未初始化则生成）
        if self.current_hidden is None:
            batch_hidden, batch_cell = self.features_extractor.init_hidden_state(batch_size=batch_size)
        else:
            # 确保隐藏状态和当前批次大小匹配
            if self.current_hidden.shape[1] != batch_size:
                batch_hidden, batch_cell = self.features_extractor.init_hidden_state(batch_size=batch_size)
            else:
                batch_hidden, batch_cell = self.current_hidden, self.current_cell

        # 步骤2：调用LSTM，保留完整时序依赖
        lstm_features, updated_hidden, updated_cell = self.features_extractor(
            obs, batch_hidden, batch_cell, return_hidden=True
        )

        # # 步骤3：更新隐藏状态（保持训练/推理一致）
        # self.current_hidden = updated_hidden
        # self.current_cell = updated_cell
        self.current_hidden = updated_hidden.detach()
        self.current_cell = updated_cell.detach()

        # 3. 后续逻辑不变（MLP提取器仅接收lstm_features张量）
        pi_features, vf_features = self.mlp_extractor(lstm_features)
        action_dist = self._get_action_dist_from_latent(pi_features)
        log_probs = action_dist.log_prob(actions)
        values = self.value_net(vf_features)
        entropy = action_dist.entropy()

        return values, log_probs, entropy

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    【消融实验3】移除LayerNorm的LSTM特征提取器（消融输入归一化设计）
    仅修改此处，其余所有代码和原模型完全一致
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 2:
            self.seq_len = observation_space.shape[0]
            self.obs_dim = observation_space.shape[1]
        else:
            self.seq_len = 11
            self.obs_dim = 64

        # 消融：移除原有的LayerNorm层，完全删除归一化模块
        # self.lstm_input_norm = nn.LayerNorm(self.obs_dim)

        # LSTM核心层和原模型完全一致
        self.lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )

    def forward(self, observations, hidden_state=None, cell_state=None, return_hidden=False):
        if len(observations.shape) == 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, self.seq_len, self.obs_dim)
        batch_size, seq_len, _ = observations.shape

        # 设备与隐藏状态初始化和原模型完全一致
        device = observations.device
        hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)

        # 消融：移除归一化，原始观测直接输入LSTM
        # normalized_obs = self.lstm_input_norm(observations)
        lstm_out, (updated_hidden, updated_cell) = self.lstm(
            observations, (hidden_state, cell_state)
        )
        final_features = lstm_out[:, -1, :]

        # 返回值逻辑和原模型完全一致，保证接口兼容
        if return_hidden:
            return final_features, updated_hidden, updated_cell
        else:
            return final_features

    def init_hidden_state(self, batch_size=1):
        # 和原模型完全一致，无任何修改
        device = next(self.parameters()).device
        hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        return hidden_state, cell_state
    
class ConstantSchedule:
    def __init__(self, value):
        """返回固定值的调度器"""
        self.value = value  # 固定学习率值（如0.0001）

    def __call__(self, progress_remaining):
        """SB3会自动传入训练进度（0~1），此处直接返回固定值"""
        return self.value
    
def main():
    rclpy.init()

    # 启动守护节点
    guard = CmdVelGuard()

    try:
        # 初始化环境
        env = RobotEnv()
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        # x1 = random.uniform(3, 5)
        # y1 = random.uniform(-4, -5)
        env.set_goal(x=x1,y=y1)
        # env = gym.wrappers.Autoreset(env)

        custom_objects = {
            "LSTMPolicy": LSTMPolicy,
            "LSTMFeatureExtractor": LSTMFeatureExtractor,
            "CustomMLPExtractor": CustomMLPExtractor
        }
        model = PPO(LSTMPolicy, env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.7,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/",device="cpu")
        # model=PPO.load("Move_robot90",env=env,custom_objects=custom_objects,policy=LSTMPolicy,tensorboard_log="./ppo_tensorboard/")
        # print(model.tensorboard_log)
        # print(model.clip_range)
        # print(model.ent_coef)
        # print(model.vf_coef)
        # print(model.lr_schedule)
        # print(model._n_updates)
        # model.clip_range = lambda progress: 0.13
        # model.ent_coef = 0.02
        # model.vf_coef = 0.55
        # # 2. 修改学习率调度器为固定值
        # model.lr_schedule = ConstantSchedule(0.0001)
        model.clip_range = lambda progress: 0.25
        model.ent_coef = 0.1
        model.vf_coef = 0.7
        # # 2. 修改学习率调度器为固定值
        model.lr_schedule = ConstantSchedule(0.0003)
        optimizer_kwargs = model.policy.optimizer.defaults
        # 关键：删除原有lr，避免重复传递
        if 'lr' in optimizer_kwargs:
            del optimizer_kwargs['lr']
        # 3. 重新初始化优化器（应用新调度器）
        model.policy.optimizer = model.policy.optimizer.__class__(
            model.policy.parameters(),
            lr=model.lr_schedule(1.0),  # 1.0表示初始进度
            **optimizer_kwargs
        )
        model.policy.reset_hidden_state(batch_size=1)
        env.model = model
        model.learn(total_timesteps=61440,log_interval=1,tb_log_name="Move_robot30")
        guard.printall()
        model.save("Move_robot30")

    finally:
        # 确保资源清理
        env.close()

if __name__ == '__main__':
    main()