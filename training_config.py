"""DQN/PPO 학습 설정을 한곳에서 조절하는 파일."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class EnvironmentConfig:
    # 브라우저/게임 환경
    render_mode: Optional[str] = None
    headless: bool = False
    initial_wait_seconds: float = 20.0
    upright_angle: float = 5.0
    terminal_angle: float = 70.0
    idle_after_terminal: float = 2.0
    upright_timeout: float = 15.0
    upright_detection_retry_seconds: float = 3.0
    upright_detection_max_retries: int = 1
    upright_fallen_retry_seconds: float = 3.0
    upright_fallen_max_retries: int = 1
    # 최초 비학습 게임이 terminal_angle을 넘을 때까지 추적하는 최대 시간
    initial_tracking_timeout: float = 120.0
    action_duration: float = 0.06
    observation_delay: float = 0.025
    max_episode_steps: int = 10_000

    # 테스트 단계에서는 True, 장시간 학습에서는 False 권장
    print_status: bool = True


@dataclass
class RewardConfig:
    # 매 스텝 생존 보상
    alive_reward: float = 5.0

    # 중심에 가까울수록 추가되는 보상
    upright_bonus: float = 3.0
    upright_bonus_angle: float = 15.0

    # 각도가 커질수록 제곱으로 증가하는 감점
    angle_penalty: float = 1.0

    # 각도 복원 방향 행동 보상
    # +각도에서는 왼쪽(0), -각도에서는 오른쪽(1)이 정답 행동입니다.
    correct_direction_reward: float = 1.0
    wrong_direction_penalty: float = -2.0
    # 0도 근처에서 좌우 보상이 빠르게 뒤집히는 것을 막는 무시 구간
    direction_dead_zone: float = 10.0

    # 각도 검출 실패 및 에피소드 종료 감점
    detection_failure_penalty: float = -0.25
    terminal_penalty: float = -10.0


@dataclass
class RunConfig:
    # 둘 중 먼저 만족하면 학습 종료
    total_timesteps: int = 100_000
    max_episodes: int = 1_000

    checkpoint_freq: int = 5_000
    seed: int = 42
    device: str = "auto"
    # Rich/tqdm 종료 오류를 피하기 위해 기본 비활성화
    progress_bar: bool = False

    # 기존 모델을 이어서 학습하려면 zip 경로 입력, 새 학습은 None
    resume_model_path: Optional[str] = None

    models_root: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    logs_root: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")


@dataclass
class DQNConfig:
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.30
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net_arch: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [128, 128])


# 주로 수정할 설정 객체들
ENV_CONFIG = EnvironmentConfig()
REWARD_CONFIG = RewardConfig()

DQN_RUN_CONFIG = RunConfig(
    total_timesteps=100_000,
    max_episodes=1_000,
    resume_model_path=r"models\dqn\20260724_173919\final_model.zip"
)
DQN_CONFIG = DQNConfig()

PPO_RUN_CONFIG = RunConfig(
    total_timesteps=100_000,
    max_episodes=1_000,
    
)
PPO_CONFIG = PPOConfig()
