"""환경 생성 코드를 DQN과 PPO에서 공유합니다."""

from pathlib import Path
from typing import Optional

from stable_baselines3.common.monitor import Monitor

from reward_wrapper import ConfigurableRewardWrapper
from stork_env import StorkGameEnv
from training_config import EnvironmentConfig, RewardConfig


def make_env(
    env_config: EnvironmentConfig,
    reward_config: RewardConfig,
    monitor_file: Optional[Path] = None,
):
    base_env = StorkGameEnv(
        render_mode=env_config.render_mode,
        headless=env_config.headless,
        initial_wait_seconds=env_config.initial_wait_seconds,
        upright_angle=env_config.upright_angle,
        terminal_angle=env_config.terminal_angle,
        idle_after_terminal=env_config.idle_after_terminal,
        upright_timeout=env_config.upright_timeout,
        upright_detection_retry_seconds=(
            env_config.upright_detection_retry_seconds
        ),
        upright_detection_max_retries=(
            env_config.upright_detection_max_retries
        ),
        upright_fallen_retry_seconds=(
            env_config.upright_fallen_retry_seconds
        ),
        upright_fallen_max_retries=(
            env_config.upright_fallen_max_retries
        ),
        initial_tracking_timeout=env_config.initial_tracking_timeout,
        action_duration=env_config.action_duration,
        observation_delay=env_config.observation_delay,
        max_episode_steps=env_config.max_episode_steps,
        print_status=env_config.print_status,
    )
    rewarded_env = ConfigurableRewardWrapper(base_env, reward_config)

    filename = str(monitor_file) if monitor_file is not None else None
    return Monitor(
        rewarded_env,
        filename=filename,
        info_keywords=(
            "episode_number",
            "angle_degrees",
            "termination_reason",
            "reward_total",
        ),
    )
