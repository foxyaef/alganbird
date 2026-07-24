"""PPO 학습 실행 파일."""

from pathlib import Path

from stable_baselines3 import PPO

from env_factory import make_env
from training_common import prepare_run
from training_config import (
    ENV_CONFIG,
    PPO_CONFIG,
    PPO_RUN_CONFIG,
    REWARD_CONFIG,
)


def main():
    model_dir, log_dir, callbacks = prepare_run(
        "ppo", PPO_RUN_CONFIG, ENV_CONFIG, REWARD_CONFIG, PPO_CONFIG
    )
    env = make_env(
        ENV_CONFIG,
        REWARD_CONFIG,
        monitor_file=log_dir / "train.monitor.csv",
    )

    resume_path = PPO_RUN_CONFIG.resume_model_path
    if resume_path:
        print(f"[PPO] 기존 모델 이어서 학습: {resume_path}", flush=True)
        model = PPO.load(
            resume_path,
            env=env,
            device=PPO_RUN_CONFIG.device,
            tensorboard_log=str(log_dir / "tensorboard"),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_CONFIG.learning_rate,
            n_steps=PPO_CONFIG.n_steps,
            batch_size=PPO_CONFIG.batch_size,
            n_epochs=PPO_CONFIG.n_epochs,
            gamma=PPO_CONFIG.gamma,
            gae_lambda=PPO_CONFIG.gae_lambda,
            clip_range=PPO_CONFIG.clip_range,
            ent_coef=PPO_CONFIG.ent_coef,
            vf_coef=PPO_CONFIG.vf_coef,
            max_grad_norm=PPO_CONFIG.max_grad_norm,
            policy_kwargs={"net_arch": PPO_CONFIG.net_arch},
            tensorboard_log=str(log_dir / "tensorboard"),
            seed=PPO_RUN_CONFIG.seed,
            device=PPO_RUN_CONFIG.device,
            verbose=1,
        )

    final_path = model_dir / "final_model"
    try:
        print(
            f"[PPO] 학습 시작: 최대 {PPO_RUN_CONFIG.total_timesteps:,} steps / "
            f"{PPO_RUN_CONFIG.max_episodes:,} episodes",
            flush=True,
        )
        model.learn(
            total_timesteps=int(PPO_RUN_CONFIG.total_timesteps),
            callback=callbacks,
            progress_bar=PPO_RUN_CONFIG.progress_bar,
            reset_num_timesteps=not bool(resume_path),
            tb_log_name="PPO",
        )
    except KeyboardInterrupt:
        print("[PPO] 사용자 중단: 현재 모델을 저장합니다.", flush=True)
    except Exception:
        print("[PPO] 학습 중 실제 오류가 발생했습니다:", flush=True)
        raise
    finally:
        model.save(str(final_path))
        env.close()

    print(f"[PPO] 최종 모델: {Path(str(final_path) + '.zip')}", flush=True)


if __name__ == "__main__":
    main()
