"""DQN/PPO가 공유하는 실행 폴더, 설정 저장, 콜백 생성 코드."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    StopTrainingOnMaxEpisodes,
)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def prepare_run(algorithm, run_config, env_config, reward_config, algo_config):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(run_config.models_root) / algorithm / run_id
    log_dir = Path(run_config.logs_root) / algorithm / run_id
    checkpoint_dir = model_dir / "checkpoints"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "algorithm": algorithm,
        "run": asdict(run_config),
        "environment": asdict(env_config),
        "reward": asdict(reward_config),
        "algorithm_hyperparameters": asdict(algo_config),
    }
    (model_dir / "config_snapshot.json").write_text(
        json.dumps(_jsonable(snapshot), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, int(run_config.checkpoint_freq)),
        save_path=str(checkpoint_dir),
        name_prefix=f"{algorithm}_stork",
        save_replay_buffer=algorithm == "dqn",
        verbose=2,
    )
    episode_callback = StopTrainingOnMaxEpisodes(
        max_episodes=int(run_config.max_episodes),
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, episode_callback])
    return model_dir, log_dir, callbacks

