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

# def main():
#     rclpy.init()

#     # 启动守护节点
#     guard = CmdVelGuard1()

#     try:
#         # 初始化环境
#         env = RobotEnv1()
#         # env.set_goal(x=5.0,y=1.5)
#         x1 = random.uniform(4, 5)
#         y1 = random.uniform(1, 2)
#         env.set_goal(x=x1,y=y1)
#         env = gym.wrappers.Autoreset(env)

#         model = PPO("MlpPolicy", env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.6,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/")
#         # model=PPO.load("PPO_robot20",env=env)
#         # print(model.tensorboard_log)
#         # print(model.num_timesteps)
#         model.clip_range = lambda progress: 0.25
#         model.ent_coef = 0.1
#         model.vf_coef = 0.6
#         # 2. 修改学习率调度器为固定值
#         model.lr_schedule = ConstantSchedule(0.0003)
#         optimizer_kwargs = model.policy.optimizer.defaults
#         # 关键：删除原有lr，避免重复传递
#         if 'lr' in optimizer_kwargs:
#             del optimizer_kwargs['lr']
#         # 3. 重新初始化优化器（应用新调度器）
#         model.policy.optimizer = model.policy.optimizer.__class__(
#             model.policy.parameters(),
#             lr=model.lr_schedule(1.0),  # 1.0表示初始进度
#             **optimizer_kwargs
#         )
#         model.learn(total_timesteps=10240,log_interval=1,tb_log_name="PPO_robot5")
#         guard.printall()
#         model.save("PPO_robot5")

#     finally:
#         # 确保资源清理
#         env.close()



    # rclpy.init()

    # # 启动守护节点
    # guard = CmdVelGuard1()

    # try:
    #     # 初始化环境
    #     env = RobotEnv1()
    #     env.set_goal(x=5.0,y=1.5)
    #     env = gym.wrappers.Autoreset(env)

        # model = DQN("MlpPolicy", env, verbose=1,replay_buffer_class=SimplePERBuffer,
        #     replay_buffer_kwargs={
        #         "alpha": 0.6,  # PER优先级权重
        #         "beta": 0.6,   # 固定beta（简化版，无需动态调整）
        #     },buffer_size=2000000)
        # model = A2C(
        #     "MlpPolicy",
        #     env,
        #     verbose=1,
        #     # 1. 学习率：降至2.5e-4，进一步降低策略震荡
        #     learning_rate=2.5e-4,
        #     # 2. 每批样本数：增至128，提升On-Policy样本利用率（关键）
        #     n_steps=128,
        #     # 3. 熵系数：增至0.2，强制探索新路径，避免徘徊超时
        #     ent_coef=0.2,
        #     # 4. 折扣因子：0.92，更聚焦即时避障+目标接近奖励
        #     gamma=0.92,
        #     # 5. 价值函数权重：0.9，提升Critic评估精度，减少价值损失
        #     vf_coef=0.9,
        #     # 6. 梯度裁剪：1.5，防止梯度爆炸，稳定策略更新
        #     max_grad_norm=1.5,
        # )
    #     model=A2C.load("A2C_robot40",env=env)
    #     model.learn(total_timesteps=40960,log_interval=20)
    #     guard.printall()
    #     model.save("A2C_robot60")

    # finally:
    #     # 确保资源清理
    #     env.close()

class SimplePERBuffer(ReplayBuffer):
    """适配旧版SB3的极简PER（无alpha/beta，无多余参数）"""
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        optimize_memory_usage=False,
        **kwargs,  # 仅接收参数，不使用
    ):
        # 父类初始化：仅传它认识的基础参数（核心！）
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage
        )
        # 仅保留优先级数组，无alpha/beta（避免参数传递错误）
        self.priorities = np.ones((buffer_size,), dtype=np.float32)
        self.max_priority = 10.0  # 新样本高优先级

    def add(self, *args, **kwargs):
        """新样本赋予最高优先级"""
        super().add(*args, **kwargs)
        self.priorities[self.pos - 1] = self.max_priority

    def sample(self, batch_size, env=None, **kwargs):
        """仅按优先级采样，返回原生结果"""
        # 1. 计算优先级概率
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = priorities / priorities.sum()

        # 2. 按优先级采样索引
        indices = np.random.choice(len(probs), batch_size, p=probs)

        # 3. 返回原生采样结果（不修改）
        return super()._get_samples(indices)




class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    重构后的LSTM特征提取器：
    1. 移除全局隐藏状态变量，改为显式输入输出（避免跨episode/批次共享）
    2. 保留LSTM梯度传播，支持时序依赖学习
    3. 提供初始隐藏状态生成方法，方便调用者初始化
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        if len(observation_space.shape) == 2:
            # 2D观测空间：(seq_len, obs_dim)（如11,64）
            self.seq_len = observation_space.shape[0]
            self.obs_dim = observation_space.shape[1]
        else:
            # 1D观测空间：展平的时序（如704=11×64），手动指定seq_len=11
            self.seq_len = 11  # 你的时序步数固定为11
            self.obs_dim = 64  # 每步观测维度固定为64

        # 输入归一化：稳定LSTM训练（避免输入值范围过大导致梯度爆炸）
        self.lstm_input_norm = nn.LayerNorm(self.obs_dim)

        # LSTM核心层：单一层、batch_first=True（输入格式：batch, seq_len, obs_dim）
        self.lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=features_dim,  # LSTM输出特征维度（与策略/价值网络匹配）
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # 单一层无需dropout，避免过拟合
            bidirectional=False  # 导航任务无需双向（只需前向时序依赖）
        )

    def forward(self, observations, hidden_state=None, cell_state=None, return_hidden=False):
        """
        新增 return_hidden 参数：
        - return_hidden=False（默认）：仅返回特征张量（兼容SB3自动调用）
        - return_hidden=True：返回 (features, updated_hidden, updated_cell)（策略主动调用）
        """
        if len(observations.shape) == 2:
        # 强制重塑为 (batch_size, 11, 64)（你的固定时序参数）
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, self.seq_len, self.obs_dim)
        # 解包3维张量（确保batch_size正确）
        batch_size, seq_len, _ = observations.shape

        # # 1. 自动生成临时隐藏状态（无传入时）（原逻辑不变）
        # if hidden_state is None or cell_state is None:
        #     device = observations.device
        #     hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        #     cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        device = observations.device
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        if cell_state is None:
            cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)

        # 2. LSTM前向计算（原逻辑不变）
        normalized_obs = self.lstm_input_norm(observations)
        lstm_out, (updated_hidden, updated_cell) = self.lstm(
            normalized_obs, (hidden_state, cell_state)
        )
        final_features = lstm_out[:, -1, :]

        # 3. 根据 return_hidden 控制返回值
        if return_hidden:
            return final_features, updated_hidden, updated_cell
        else:
            return final_features  # 默认仅返回特征张量，兼容SB3

    def init_hidden_state(self, batch_size=1):
        """
        生成初始隐藏状态（全0），供调用者（如Policy）初始化使用
        Args:
            batch_size: 批次大小（单环境=1，多环境=环境数量）
        Returns:
            hidden_state: 初始隐藏状态，shape=(1, batch_size, features_dim)
            cell_state: 初始细胞状态，shape=(1, batch_size, features_dim)
        """
        device = next(self.parameters()).device  # 与模型同设备（CPU/GPU）
        hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)
        return hidden_state, cell_state

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

# class LSTMFeatureExtractor(BaseFeaturesExtractor):
#     """
#     【消融实验3】移除LayerNorm的LSTM特征提取器（消融输入归一化设计）
#     仅修改此处，其余所有代码和原模型完全一致
#     """
#     def __init__(self, observation_space, features_dim=64):
#         super().__init__(observation_space, features_dim)
#         if len(observation_space.shape) == 2:
#             self.seq_len = observation_space.shape[0]
#             self.obs_dim = observation_space.shape[1]
#         else:
#             self.seq_len = 11
#             self.obs_dim = 64

#         # 消融：移除原有的LayerNorm层，完全删除归一化模块
#         # self.lstm_input_norm = nn.LayerNorm(self.obs_dim)

#         # LSTM核心层和原模型完全一致
#         self.lstm = nn.LSTM(
#             input_size=self.obs_dim,
#             hidden_size=features_dim,
#             num_layers=1,
#             batch_first=True,
#             dropout=0.0,
#             bidirectional=False
#         )

#     def forward(self, observations, hidden_state=None, cell_state=None, return_hidden=False):
#         if len(observations.shape) == 2:
#             batch_size = observations.shape[0]
#             observations = observations.reshape(batch_size, self.seq_len, self.obs_dim)
#         batch_size, seq_len, _ = observations.shape

#         # 设备与隐藏状态初始化和原模型完全一致
#         device = observations.device
#         hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
#         cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)

#         # 消融：移除归一化，原始观测直接输入LSTM
#         # normalized_obs = self.lstm_input_norm(observations)
#         lstm_out, (updated_hidden, updated_cell) = self.lstm(
#             observations, (hidden_state, cell_state)
#         )
#         final_features = lstm_out[:, -1, :]

#         # 返回值逻辑和原模型完全一致，保证接口兼容
#         if return_hidden:
#             return final_features, updated_hidden, updated_cell
#         else:
#             return final_features

#     def init_hidden_state(self, batch_size=1):
#         # 和原模型完全一致，无任何修改
#         device = next(self.parameters()).device
#         hidden_state = torch.zeros(1, batch_size, self.features_dim, device=device)
#         cell_state = torch.zeros(1, batch_size, self.features_dim, device=device)
#         return hidden_state, cell_state




class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super().__init__()

        # 论文结构：输入层 LayerNorm（保留）
        self.input_norm = nn.LayerNorm(input_dim)

        # -------------------------- 论文原生核心结构（仅改双向→单向） --------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,  # ✅ 改1：双向→单向（更适合导航）
            batch_first=True,
            dropout=0.0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)  # ✅ 改2：去掉 *2

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # ✅ 改3：去掉 *2
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)  # ✅ 改4：去掉 *2

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # ✅ 改5：输入维度去掉 *2
            nn.GELU()
        )
        # ---------------------------------------------------------------------------

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        # ✅ 改6：单向LSTM初始化层数 2→1
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)

        # 论文结构：输入归一化（保留）
        x = self.input_norm(x)

        # -------------------------- 论文原生核心前向逻辑（保留） --------------------------
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_norm(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.attn_norm(lstm_out + attn_out)  # ✅ 论文残差连接（保留）

        # 取最后一步输出（保留之前的修改，更适合导航）
        out = self.proj(out)[:, -1, :]
        # ---------------------------------------------------------------------------
        return out

# ===========================================================================
# 2. 特征提取器（完全保留）
# ===========================================================================
class BiLSTM_Attention_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        if len(observation_space.shape) == 2:
            self.seq_len = observation_space.shape[0]
            self.obs_dim = observation_space.shape[1]
        else:
            self.seq_len = 11
            self.obs_dim = observation_space.shape[-1] // self.seq_len

        input_dim = self.obs_dim
        super().__init__(observation_space, features_dim)
        self.model = BiLSTM_Attention(input_dim=input_dim, hidden_dim=features_dim)

    def forward(self, observations):
        if len(observations.shape) == 2:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, self.seq_len, self.obs_dim)
        return self.model(observations)

# ===========================================================================
# 3. 极简通用分离 MLP（完全保留）
# ===========================================================================
class MinimalSeparateMLP(nn.Module):
    def __init__(self, features_dim=64):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.vf = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

    def forward(self, features):
        return self.pi(features), self.vf(features)

    def forward_actor(self, features):
        return self.pi(features)

    def forward_critic(self, features):
        return self.vf(features)

# ===========================================================================
# 4. 策略类（完全保留）
# ===========================================================================
class BiLSTM_Timing_Policy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=BiLSTM_Attention_FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=None,
            activation_fn=nn.GELU,
            **kwargs
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = MinimalSeparateMLP(features_dim=self.features_dim)

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
        model = PPO(LSTMPolicy, env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.7,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/",device="cuda")
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

# def main():
#     rclpy.init()

#     # 启动守护节点
#     guard = CmdVelGuard()

#     try:
#         # 初始化环境
#         env = RobotEnv()
#         # env.set_goal(x=5.0,y=1.5)
#         x1 = random.uniform(4, 5)
#         y1 = random.uniform(1, 2)
#         env.set_goal(x=x1,y=y1)
#         env = gym.wrappers.Autoreset(env)
#         model = PPO(
#         BiLSTM_Timing_Policy,
#         env,
#         learning_rate=0.0003,
#         clip_range=0.25,
#         ent_coef=0.1,
#         vf_coef=0.6,
#         verbose=1,
#         tensorboard_log="./ppo_tensorboard/"
#     )
#         # model=PPO.load("A2C_robot30",env=env)
#         model.clip_range = lambda progress: 0.25
#         model.ent_coef = 0.1
#         model.vf_coef = 0.6
#         # 2. 修改学习率调度器为固定值
#         model.lr_schedule = ConstantSchedule(0.0003)
#         optimizer_kwargs = model.policy.optimizer.defaults
#         # 关键：删除原有lr，避免重复传递
#         if 'lr' in optimizer_kwargs:
#             del optimizer_kwargs['lr']
#         # 3. 重新初始化优化器（应用新调度器）
#         model.policy.optimizer = model.policy.optimizer.__class__(
#             model.policy.parameters(),
#             lr=model.lr_schedule(1.0),  # 1.0表示初始进度
#             **optimizer_kwargs
#         )
#         model.learn(total_timesteps=10240,log_interval=1,tb_log_name="BiLSTM_robot5")
#         guard.printall()
#         model.save("BiLSTM_robot5")

#     finally:
#         # 确保资源清理
#         env.close()

if __name__ == '__main__':
    main()