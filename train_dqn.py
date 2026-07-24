"""DQN 학습 실행 파일."""

from pathlib import Path

from stable_baselines3 import DQN

from env_factory import make_env
from training_common import prepare_run
from training_config import (
    DQN_CONFIG,
    DQN_RUN_CONFIG,
    ENV_CONFIG,
    REWARD_CONFIG,
)


def main():
    model_dir, log_dir, callbacks = prepare_run(
        "dqn", DQN_RUN_CONFIG, ENV_CONFIG, REWARD_CONFIG, DQN_CONFIG
    )
    env = make_env(
        ENV_CONFIG,
        REWARD_CONFIG,
        monitor_file=log_dir / "train.monitor.csv",
    )

    resume_path = DQN_RUN_CONFIG.resume_model_path
    if resume_path:
        print(f"[DQN] 기존 모델 이어서 학습: {resume_path}", flush=True)
        model = DQN.load(
            resume_path,
            env=env,
            device=DQN_RUN_CONFIG.device,
            tensorboard_log=str(log_dir / "tensorboard"),
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=DQN_CONFIG.learning_rate,
            buffer_size=DQN_CONFIG.buffer_size,
            learning_starts=DQN_CONFIG.learning_starts,
            batch_size=DQN_CONFIG.batch_size,
            tau=DQN_CONFIG.tau,
            gamma=DQN_CONFIG.gamma,
            train_freq=DQN_CONFIG.train_freq,
            gradient_steps=DQN_CONFIG.gradient_steps,
            target_update_interval=DQN_CONFIG.target_update_interval,
            exploration_fraction=DQN_CONFIG.exploration_fraction,
            exploration_initial_eps=DQN_CONFIG.exploration_initial_eps,
            exploration_final_eps=DQN_CONFIG.exploration_final_eps,
            policy_kwargs={"net_arch": DQN_CONFIG.net_arch},
            tensorboard_log=str(log_dir / "tensorboard"),
            seed=DQN_RUN_CONFIG.seed,
            device=DQN_RUN_CONFIG.device,
            verbose=1,
        )

    final_path = model_dir / "final_model"
    try:
        print(
            f"[DQN] 학습 시작: 최대 {DQN_RUN_CONFIG.total_timesteps:,} steps / "
            f"{DQN_RUN_CONFIG.max_episodes:,} episodes",
            flush=True,
        )
        model.learn(
            total_timesteps=int(DQN_RUN_CONFIG.total_timesteps),
            callback=callbacks,
            progress_bar=DQN_RUN_CONFIG.progress_bar,
            reset_num_timesteps=not bool(resume_path),
            tb_log_name="DQN",
        )
    except KeyboardInterrupt:
        print("[DQN] 사용자 중단: 현재 모델을 저장합니다.", flush=True)
    except Exception:
        print("[DQN] 학습 중 실제 오류가 발생했습니다:", flush=True)
        raise
    finally:
        model.save(str(final_path))
        model.save_replay_buffer(str(model_dir / "final_replay_buffer.pkl"))
        env.close()

    print(f"[DQN] 최종 모델: {Path(str(final_path) + '.zip')}", flush=True)


if __name__ == "__main__":
    main()
