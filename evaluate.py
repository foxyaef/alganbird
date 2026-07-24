"""저장된 DQN/PPO 모델을 실제 게임에서 평가합니다."""

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO

from env_factory import make_env
from training_config import ENV_CONFIG, REWARD_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--model", required=True, help="저장된 model.zip 경로")
    parser.add_argument("--episodes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    ENV_CONFIG.print_status = True
    ENV_CONFIG.render_mode = "human"
    env = make_env(ENV_CONFIG, REWARD_CONFIG)
    algorithm_class = DQN if args.algorithm == "dqn" else PPO
    model = algorithm_class.load(str(model_path), env=env)

    rewards = []
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            total_reward = 0.0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    rewards.append(total_reward)
                    print(
                        f"[평가 {episode}/{args.episodes}] "
                        f"reward={total_reward:.2f}, "
                        f"steps={info['episode_steps']}, "
                        f"reason={info['termination_reason']}",
                        flush=True,
                    )
                    break
    finally:
        env.close()

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[평가 완료] 평균 reward={mean_reward:.2f}", flush=True)


if __name__ == "__main__":
    main()

