import random
import os

from stable_baselines3 import PPO
from fishbot_application.robot_env import RobotEnv, CmdVelGuard
from fishbot_application.robot_env1 import RobotEnv1, CmdVelGuard1
from fishbot_application.train import LSTMPolicy, LSTMFeatureExtractor, CustomMLPExtractor
from fishbot_application.train import BiLSTM_Timing_Policy
import gymnasium as gym
import rclpy
import time

MODEL_DIR = "/home/s-lin/LIN/ros2_ws/chapt6_ws"


def predict_one(env_class, guard_class, model_path, model_name,
                policy=None, custom_objects=None, total_timesteps=20000):
    """运行单个模型的预测，返回 (model_name, stats)"""
    print(f"\n{'='*60}")
    print(f"Predicting: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    rclpy.init()
    guard = guard_class()
    env = env_class()
    env.reset_episode_counter()
    env.reset_stats()

    try:
        x1 = random.uniform(4, 5)
        y1 = random.uniform(1, 2)
        env.set_goal(x=x1, y=y1)

        if custom_objects and policy:
            model = PPO.load(model_path, env=env,
                             custom_objects=custom_objects, policy=policy)
        else:
            model = PPO.load(model_path, env=env)

        model.policy.set_training_mode(False)
        env.model = model

        obs, _ = env.reset()
        for step in range(total_timesteps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, _ = env.reset()

        guard.printall()
        return (model_name, env.get_stats())

    finally:
        env.close()


def _print_summary_table(all_results):
    """打印所有模型的预测汇总对比表"""
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)

    header = f"{'Model':<20} {'Episodes':>9} {'Success%':>9} {'AvgReward':>10} {'AvgSteps':>9} {'AvgDist':>8}"
    print(header)
    print("-" * len(header))

    for model_name, stats in all_results:
        total = len(stats)
        if total == 0:
            print(f"{model_name:<20} {'(no data)':>50}")
            continue

        arrives = sum(1 for s in stats if s['event'] == 'arrive')
        success_rate = arrives / total * 100
        avg_reward = sum(s['reward'] for s in stats) / total
        avg_steps = sum(s['steps'] for s in stats) / total
        avg_dist = sum(s['goal_dist'] for s in stats) / total

        print(f"{model_name:<20} {total:>9} {success_rate:>8.1f}% "
              f"{avg_reward:>10.2f} {avg_steps:>9.1f} {avg_dist:>8.2f}")

    print("-" * len(header))
    print("=" * 80)


def main():
    """单模型预测 — 默认 LSTM Move_robot30"""
    custom_objects = {
        "LSTMPolicy": LSTMPolicy,
        "LSTMFeatureExtractor": LSTMFeatureExtractor,
        "CustomMLPExtractor": CustomMLPExtractor
    }
    predict_one(
        env_class=RobotEnv,
        guard_class=CmdVelGuard,
        model_path=os.path.join(MODEL_DIR, "Move_robot30.zip"),
        model_name="LSTM (Move_robot30)",
        policy=LSTMPolicy,
        custom_objects=custom_objects,
    )


def predict_all():
    """依次测试全部 3 个算法模型 + 打印汇总表"""
    all_results = []
    lstm_custom = {
        "LSTMPolicy": LSTMPolicy,
        "LSTMFeatureExtractor": LSTMFeatureExtractor,
        "CustomMLPExtractor": CustomMLPExtractor
    }

    # [1/3] LSTM
    all_results.append(predict_one(
        env_class=RobotEnv,
        guard_class=CmdVelGuard,
        model_path=os.path.join(MODEL_DIR, "Move_robot30.zip"),
        model_name="[1/3] LSTM",
        policy=LSTMPolicy,
        custom_objects=lstm_custom,
    ))

    # [2/3] PPO
    all_results.append(predict_one(
        env_class=RobotEnv1,
        guard_class=CmdVelGuard1,
        model_path=os.path.join(MODEL_DIR, "PPO_robot5.zip"),
        model_name="[2/3] PPO",
    ))

    # [3/3] BiLSTM
    all_results.append(predict_one(
        env_class=RobotEnv,
        guard_class=CmdVelGuard,
        model_path=os.path.join(MODEL_DIR, "BiLSTM_robot5.zip"),
        model_name="[3/3] BiLSTM",
        policy=BiLSTM_Timing_Policy,
        custom_objects=lstm_custom,
    ))

    print(f"\n{'='*60}")
    print("All predictions completed!")
    _print_summary_table(all_results)


if __name__ == '__main__':
    main()
