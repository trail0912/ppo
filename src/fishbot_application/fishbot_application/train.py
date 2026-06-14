import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
import torch
from fishbot_application.robot_env import RobotEnv,CmdVelGuard
from fishbot_application.robot_env1 import RobotEnv1,CmdVelGuard1
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

def lstm_main():
    rclpy.init()

    # 启动守护节点
    guard = CmdVelGuard()

    try:
        # 初始化环境
        env = RobotEnv()
        env.reset_episode_counter()
        env.reset_stats()
        raw_env = env  # 保存引用，用于最后取 stats
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        env.set_goal(x=x1,y=y1)

        custom_objects = {
            "LSTMPolicy": LSTMPolicy,
            "LSTMFeatureExtractor": LSTMFeatureExtractor,
            "CustomMLPExtractor": CustomMLPExtractor
        }
        model = PPO(LSTMPolicy, env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.7,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/",device="auto")
        model.clip_range = lambda progress: 0.25
        model.ent_coef = 0.1
        model.vf_coef = 0.7
        model.lr_schedule = ConstantSchedule(0.0003)
        optimizer_kwargs = model.policy.optimizer.defaults
        if 'lr' in optimizer_kwargs:
            del optimizer_kwargs['lr']
        model.policy.optimizer = model.policy.optimizer.__class__(
            model.policy.parameters(),
            lr=model.lr_schedule(1.0),
            **optimizer_kwargs
        )
        model.policy.reset_hidden_state(batch_size=1)
        env.model = model
        model.learn(total_timesteps=61440,log_interval=1,tb_log_name="Move_robot30")
        guard.printall()
        model.save("Move_robot30")
        return ("LSTM", raw_env.get_stats())

    finally:
        env.close()

def bilstm_main():
    rclpy.init()

    # 启动守护节点
    guard = CmdVelGuard()

    try:
        # 初始化环境
        env = RobotEnv()
        env.reset_episode_counter()
        env.reset_stats()
        raw_env = env
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        env.set_goal(x=x1,y=y1)
        env = gym.wrappers.Autoreset(env)

        model = PPO(BiLSTM_Timing_Policy, env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.6,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/")
        model.clip_range = lambda progress: 0.25
        model.ent_coef = 0.1
        model.vf_coef = 0.6
        model.lr_schedule = ConstantSchedule(0.0003)
        optimizer_kwargs = model.policy.optimizer.defaults
        if 'lr' in optimizer_kwargs:
            del optimizer_kwargs['lr']
        model.policy.optimizer = model.policy.optimizer.__class__(
            model.policy.parameters(),
            lr=model.lr_schedule(1.0),
            **optimizer_kwargs
        )
        model.learn(total_timesteps=61440,log_interval=1,tb_log_name="BiLSTM_robot5")
        guard.printall()
        model.save("BiLSTM_robot5")
        return ("BiLSTM", raw_env.get_stats())

    finally:
        env.close()

def ppo_main():
    rclpy.init()

    # 启动守护节点
    guard = CmdVelGuard1()

    try:
        # 初始化环境
        env = RobotEnv1()
        env.reset_episode_counter()
        env.reset_stats()
        raw_env = env
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        env.set_goal(x=x1,y=y1)
        env = gym.wrappers.Autoreset(env)

        model = PPO("MlpPolicy", env, verbose=1,clip_range=0.25, ent_coef=0.1,vf_coef=0.6,learning_rate=0.0003,tensorboard_log="./ppo_tensorboard/")
        model.clip_range = lambda progress: 0.25
        model.ent_coef = 0.1
        model.vf_coef = 0.6
        model.lr_schedule = ConstantSchedule(0.0003)
        optimizer_kwargs = model.policy.optimizer.defaults
        if 'lr' in optimizer_kwargs:
            del optimizer_kwargs['lr']
        model.policy.optimizer = model.policy.optimizer.__class__(
            model.policy.parameters(),
            lr=model.lr_schedule(1.0),
            **optimizer_kwargs
        )
        model.learn(total_timesteps=61440,log_interval=1,tb_log_name="PPO_robot5")
        guard.printall()
        model.save("PPO_robot5")
        return ("PPO", raw_env.get_stats())

    finally:
        env.close()

def _print_summary_table(all_results):
    """打印所有模型的汇总对比表"""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    header = f"{'Model':<14} {'Episodes':>9} {'Success%':>9} {'AvgReward':>10} {'AvgSteps':>9} {'AvgDist':>8}"
    print(header)
    print("-" * len(header))

    for model_name, stats in all_results:
        total = len(stats)
        if total == 0:
            print(f"{model_name:<14} {'(no data)':>50}")
            continue

        arrives = sum(1 for s in stats if s['event'] == 'arrive')
        success_rate = arrives / total * 100
        avg_reward = sum(s['reward'] for s in stats) / total
        avg_steps = sum(s['steps'] for s in stats) / total
        avg_dist = sum(s['goal_dist'] for s in stats) / total

        print(f"{model_name:<14} {total:>9} {success_rate:>8.1f}% "
              f"{avg_reward:>10.2f} {avg_steps:>9.1f} {avg_dist:>8.2f}")

    print("-" * len(header))
    print("=" * 80)


def main():
    """依次运行 3 个算法 + 2 个消融实验"""
    from fishbot_application.train1 import main as ablation1
    from fishbot_application.train2 import main as ablation2

    all_results = []

    print("=" * 60)
    print("[1/5] LSTM Training")
    print("=" * 60)
    all_results.append(lstm_main())

    print("=" * 60)
    print("[2/5] PPO Training")
    print("=" * 60)
    all_results.append(ppo_main())

    print("=" * 60)
    print("[3/5] BiLSTM Training")
    print("=" * 60)
    all_results.append(bilstm_main())

    print("=" * 60)
    print("[4/5] Ablation Study 1")
    print("=" * 60)
    all_results.append(ablation1())

    print("=" * 60)
    print("[5/5] Ablation Study 2")
    print("=" * 60)
    all_results.append(ablation2())

    print("=" * 60)
    print("All 5 training runs completed!")
    _print_summary_table(all_results)


if __name__ == '__main__':
    lstm_main()
    # bilstm_main()
    # ppo_main()