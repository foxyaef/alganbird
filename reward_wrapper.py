"""설정 파일로 보상 함수를 조절하기 위한 Gymnasium 래퍼."""

import gymnasium as gym

from training_config import RewardConfig


class ConfigurableRewardWrapper(gym.Wrapper):
    def __init__(self, env, config: RewardConfig):
        super().__init__(env)
        self.config = config

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)

        signed_angle = float(info.get("angle_degrees", 0.0))
        angle = abs(signed_angle)
        detected = bool(info.get("angle_detected", False))
        action_number = int(action)
        terminal_angle = max(float(self.env.unwrapped.terminal_angle), 1e-6)
        bonus_angle = max(float(self.config.upright_bonus_angle), 1e-6)

        alive = self.config.alive_reward
        upright_factor = max(0.0, 1.0 - angle / bonus_angle)
        upright_bonus = self.config.upright_bonus * upright_factor
        angle_penalty = self.config.angle_penalty * (angle / terminal_angle) ** 2
        detection_penalty = 0.0 if detected else self.config.detection_failure_penalty
        terminal_penalty = self.config.terminal_penalty if terminated else 0.0

        # action 0=왼쪽, action 1=오른쪽, action 2=무행동
        # 양수 각도는 왼쪽, 음수 각도는 오른쪽, 중심 근처는 무행동을 유도합니다.
        direction_reward = 0.0
        direction_correct = None
        if detected:
            if angle <= self.config.direction_dead_zone:
                correct_action = 2
            else:
                correct_action = 0 if signed_angle > 0.0 else 1
            direction_correct = action_number == correct_action
            direction_reward = (
                self.config.correct_direction_reward
                if direction_correct
                else self.config.wrong_direction_penalty
            )

        reward = (
            alive
            + upright_bonus
            - angle_penalty
            + direction_reward
            + detection_penalty
            + terminal_penalty
        )

        info.update(
            {
                "reward_alive": float(alive),
                "reward_upright_bonus": float(upright_bonus),
                "reward_angle_penalty": float(-angle_penalty),
                "reward_direction": float(direction_reward),
                "direction_correct": direction_correct,
                "reward_detection_penalty": float(detection_penalty),
                "reward_terminal_penalty": float(terminal_penalty),
                "reward_total": float(reward),
            }
        )
        return observation, float(reward), terminated, truncated, info
