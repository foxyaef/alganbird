"""cv_test.py의 목 각도 검출을 사용하는 황새 게임 Gymnasium 환경."""

import math
import time
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import sync_playwright


GAME_URL = "https://vidkidz.tistory.com/2825"
BIRD_ROI = (0.20, 0.15, 0.90, 0.96)


def detect_neck_angle(frame):
    """cv_test.py와 동일한 방법으로 현재 프레임의 목 각도를 검출합니다."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    black_lines = cv2.bitwise_not(thresh)
    close_size = max(3, int(round(height * 0.005)))
    if close_size % 2 == 0:
        close_size += 1
    black_lines = cv2.morphologyEx(
        black_lines,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), dtype=np.uint8),
    )

    left = int(width * BIRD_ROI[0])
    top = int(height * BIRD_ROI[1])
    right = int(width * BIRD_ROI[2])
    bottom = int(height * BIRD_ROI[3])
    roi = black_lines[top:bottom, left:right]

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 360,
        threshold=max(20, int(height * 0.025)),
        minLineLength=max(35, int(height * 0.08)),
        maxLineGap=max(12, int(height * 0.03)),
    )
    if lines is None:
        return None, None, thresh, None

    candidates = []
    for roi_x1, roi_y1, roi_x2, roi_y2 in lines[:, 0]:
        x1, y1 = int(roi_x1 + left), int(roi_y1 + top)
        x2, y2 = int(roi_x2 + left), int(roi_y2 + top)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        raw_angle = math.degrees(math.atan2(dx, -dy))
        angle = (raw_angle + 90) % 180 - 90

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        center_distance = math.hypot(
            (mid_x - width * 0.53) * 0.35,
            (mid_y - height * 0.60) * 0.20,
        )
        score = length - center_distance
        candidates.append((score, length, angle, (x1, y1, x2, y2)))

    _, _, angle, line = max(candidates, key=lambda item: item[0])
    return angle, line, thresh, None


class StorkGameEnv(gym.Env):
    """
    행동: 0=왼쪽 방향키, 1=오른쪽 방향키, 2=아무것도 하지 않음
    관측: [현재 각도/90, 각도 변화량/90, 검출 성공 여부]

    상태 흐름
      최초: 25초 대기 -> 준비 Space -> |각도|<=5 확인 -> 시작 Space
      종료: |각도|>60 -> 조작 중단 -> 1초 대기 -> 준비 Space
      다음: |각도|<=5 확인 -> 시작 Space
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        render_mode="human",
        headless=False,
        initial_wait_seconds=25.0,
        upright_angle=5.0,
        terminal_angle=60.0,
        idle_after_terminal=3.0,
        upright_timeout=15.0,
        upright_detection_retry_seconds=3.0,
        upright_detection_max_retries=1,
        upright_fallen_retry_seconds=3.0,
        upright_fallen_max_retries=1,
        initial_tracking_timeout=120.0,
        action_duration=0.06,
        observation_delay=0.025,
        max_episode_steps=10_000,
        print_status=True,
    ):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode는 None, 'human', 'rgb_array' 중 하나입니다.")

        self.render_mode = render_mode
        self.headless = bool(headless)
        self.initial_wait_seconds = float(initial_wait_seconds)
        self.upright_angle = float(upright_angle)
        self.terminal_angle = float(terminal_angle)
        self.idle_after_terminal = float(idle_after_terminal)
        self.upright_timeout = float(upright_timeout)
        self.upright_detection_retry_seconds = float(
            upright_detection_retry_seconds
        )
        self.upright_detection_max_retries = int(
            upright_detection_max_retries
        )
        self.upright_fallen_retry_seconds = float(
            upright_fallen_retry_seconds
        )
        self.upright_fallen_max_retries = int(
            upright_fallen_max_retries
        )
        self.initial_tracking_timeout = float(initial_tracking_timeout)
        self.action_duration = float(action_duration)
        self.observation_delay = float(observation_delay)
        self.max_episode_steps = int(max_episode_steps)
        self.print_status = bool(print_status)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._playwright = None
        self._browser = None
        self._page = None
        self._last_frame = None
        self._last_line = None
        self._angle_history = deque(maxlen=5)
        self._previous_angle = 0.0
        self._episode_number = 0
        self._episode_steps = 0
        self._episode_active = False
        self._ready_space_pressed = False
        self._needs_full_reload = False
        self._initial_untrained_game_running = False

        self._launch_and_prepare_first_game()

    def _log(self, message):
        if self.print_status:
            print(message, flush=True)

    def _launch_and_prepare_first_game(self):
        self._log("[초기화] 브라우저를 실행합니다.")
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--window-size=960,540"],
            )
            self._page = self._browser.new_page(
                viewport={"width": 960, "height": 540}
            )
            self._page.goto(GAME_URL)
            self._wait_initial_loading()
            self._page.mouse.click(480, 270)
            self._press_ready_space("에피소드 1 준비")
            self._log("[초기화] 화면 중앙 클릭 완료")
            time.sleep(1.0)
            self._press_ready_space("에피소드 1 준비")
            self._angle_history.clear()
            self._previous_angle = 0.0
            self._initial_untrained_game_running = True
            self._log(
                "[비학습 워밍업 시작] 이 게임은 학습 에피소드에 포함하지 않습니다."
            )
        except Exception:
            self.close()
            raise

    def _wait_initial_loading(self):
        total = max(0, int(math.ceil(self.initial_wait_seconds)))
        if total == 0:
            return
        self._log(f"[초기화] 게임 로딩을 위해 {self.initial_wait_seconds:.0f}초 기다립니다.")
        started = time.monotonic()
        last_remaining = None
        while True:
            elapsed = time.monotonic() - started
            remaining = max(0, int(math.ceil(self.initial_wait_seconds - elapsed)))
            if remaining != last_remaining and (remaining % 5 == 0 or remaining <= 3):
                self._log(f"[초기화] 남은 시간: {remaining}초")
            last_remaining = remaining
            if elapsed >= self.initial_wait_seconds:
                break
            time.sleep(min(0.25, self.initial_wait_seconds - elapsed))

    def _press_ready_space(self, reason):
        self._page.keyboard.press("Space")
        self._ready_space_pressed = True
        self._log(f"[{reason}] 준비 Space 입력 완료")

    def _finish_upright_error_and_prepare_next_episode(self, reason):
        """직립 오류를 종료로 처리하고 기존 다음 에피소드 준비 순서를 실행합니다."""
        self._episode_active = False
        self._log(f"[직립 오류로 에피소드 종료] {reason}")
        self._log(
            f"[조작 정지] {self.idle_after_terminal:.2f}초 동안 "
            "어떤 키도 누르지 않습니다."
        )
        time.sleep(self.idle_after_terminal)
        self._press_ready_space("직립 오류 종료 후 초기화")
        self._log(
            f"[다음 에피소드 준비] |각도| <= "
            f"{self.upright_angle:.1f}도 확인 단계로 이동합니다."
        )
        time.sleep(self.idle_after_terminal)
        # 기존 코드의 준비 Space입니다. 이후 _wait_until_upright()가
        # 직립을 확인하고 reset()에서 Space를 한 번 더 눌러 시작합니다.
        self._press_ready_space("시작 스페이스 입력")
        self._angle_history.clear()
        self._previous_angle = 0.0
        self._last_line = None

    def _capture_frame(self):
        # cv_test.py와 동일하게 페이지 전체를 캡처합니다.
        screenshot = self._page.screenshot()
        image = np.frombuffer(screenshot, dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("스크린샷을 OpenCV 프레임으로 변환하지 못했습니다.")
        self._last_frame = frame
        return frame

    def _unwrap_angle(self, angle):
        reference = (
            self._angle_history[-1]
            if self._angle_history
            else self._previous_angle
        )
        while angle - reference > 90:
            angle -= 180
        while angle - reference < -90:
            angle += 180
        return float(angle)

    def _observe(self, state_label):
        frame = self._capture_frame()
        detected_angle, line, _, _ = detect_neck_angle(frame)
        detected = detected_angle is not None
        display_episode = (
            self._episode_number
            if self._episode_active
            else self._episode_number + 1
        )

        if detected:
            current_angle = self._unwrap_angle(detected_angle)
            angle_delta = current_angle - self._previous_angle
            self._angle_history.append(current_angle)
            smooth_angle = float(np.median(self._angle_history))
            self._previous_angle = current_angle
            self._last_line = line
            self._log(
                f"[{state_label}] episode={display_episode} "
                f"step={self._episode_steps} | 현재 각도={current_angle:+7.2f}도 | "
                f"평활 각도={smooth_angle:+7.2f}도 | 변화량={angle_delta:+7.2f}도"
            )
        else:
            current_angle = self._previous_angle
            smooth_angle = self._previous_angle
            angle_delta = 0.0
            self._log(
                f"[{state_label}] episode={display_episode} "
                f"step={self._episode_steps} | 각도 검출 실패 | "
                f"마지막 각도={current_angle:+7.2f}도"
            )

        observation = np.array(
            [
                np.clip(current_angle / 90.0, -1.0, 1.0),
                np.clip(angle_delta / 90.0, -1.0, 1.0),
                1.0 if detected else 0.0,
            ],
            dtype=np.float32,
        )
        info = {
            "episode_number": display_episode,
            "episode_steps": self._episode_steps,
            "angle_degrees": float(current_angle),
            "smoothed_angle_degrees": float(smooth_angle),
            "angle_delta_degrees": float(angle_delta),
            "angle_detected": bool(detected),
            "state": state_label,
        }
        return observation, info

    def _wait_until_upright(self):
        self._log(
            f"[직립 대기] |각도| <= {self.upright_angle:.1f}도가 될 때까지 기다립니다."
        )
        deadline = time.monotonic() + self.upright_timeout
        last_info = None
        detection_failure_started = None
        fallen_started = None
        detection_retry_count = 0
        fallen_retry_count = 0
        while time.monotonic() < deadline:
            observation, info = self._observe("직립 대기")
            last_info = info
            if info["angle_detected"] and abs(info["angle_degrees"]) <= self.upright_angle:
                self._log(
                    f"[직립 확인] 현재 각도 {info['angle_degrees']:+.2f}도 "
                    f"-> 시작 Space를 누릅니다."
                )
                return observation, info

            if info["angle_detected"]:
                detection_failure_started = None

                # 직립 대기 중인데 이미 종료 각도를 넘은 자세가 계속 보이면
                # 게임 화면이 넘어진 상태에 고정된 것이므로 Space로 복구합니다.
                if abs(info["angle_degrees"]) > self.terminal_angle:
                    now = time.monotonic()
                    if fallen_started is None:
                        fallen_started = now

                    fallen_seconds = now - fallen_started
                    can_retry_fallen = (
                        self.upright_fallen_retry_seconds > 0.0
                        and fallen_retry_count
                        < self.upright_fallen_max_retries
                    )
                    if (
                        can_retry_fallen
                        and fallen_seconds
                        >= self.upright_fallen_retry_seconds
                    ):
                        fallen_retry_count += 1
                        self._finish_upright_error_and_prepare_next_episode(
                            f"종료 각도 {info['angle_degrees']:+.2f}도가 "
                            f"{fallen_seconds:.1f}초 지속"
                        )
                        deadline = time.monotonic() + self.upright_timeout
                        self._log(
                            "[직립 복구] "
                            f"종료 각도 {info['angle_degrees']:+.2f}도가 "
                            f"{fallen_seconds:.1f}초 지속 -> "
                            f"다음 에피소드 준비 완료 ({fallen_retry_count}/"
                            f"{self.upright_fallen_max_retries})"
                        )
                        fallen_started = time.monotonic()
                else:
                    # 각도는 검출되고 있으며 아직 자연스럽게 직립 상태로
                    # 돌아오는 중인 5~60도 구간에서는 Space를 누르지 않습니다.
                    fallen_started = None
            else:
                fallen_started = None
                now = time.monotonic()
                if detection_failure_started is None:
                    detection_failure_started = now

                failure_seconds = now - detection_failure_started
                can_retry = (
                    self.upright_detection_retry_seconds > 0.0
                    and detection_retry_count
                    < self.upright_detection_max_retries
                )
                if (
                    can_retry
                    and failure_seconds
                    >= self.upright_detection_retry_seconds
                ):
                    detection_retry_count += 1
                    self._finish_upright_error_and_prepare_next_episode(
                        f"각도 미검출 {failure_seconds:.1f}초 지속"
                    )
                    deadline = time.monotonic() + self.upright_timeout
                    self._log(
                        "[직립 복구] "
                        f"각도 미검출 {failure_seconds:.1f}초 지속 -> "
                        f"다음 에피소드 준비 완료 ({detection_retry_count}/"
                        f"{self.upright_detection_max_retries})"
                    )
                    detection_failure_started = time.monotonic()
            time.sleep(0.05)

        raise RuntimeError(
            f"{self.upright_timeout:.1f}초 안에 |각도| <= "
            f"{self.upright_angle:.1f} 조건을 만족하지 못했습니다. "
            f"마지막 상태: {last_info}"
        )

    def _reload_from_beginning(self):
        self._log("[강제 초기화] 게임 페이지를 다시 불러옵니다.")
        self._page.reload()
        self._wait_initial_loading()
        self._page.mouse.click(480, 270)
        time.sleep(1.0)
        self._press_ready_space("강제 초기화")

    def _finish_initial_untrained_game(self):
        """첫 게임은 행동 없이 종료 각도 초과까지 관찰한 뒤 학습 준비 상태로 만듭니다."""
        self._log(
            "[비학습 워밍업 추적] 모델 준비 완료. 방향키를 누르지 않고 "
            f"|각도| > {self.terminal_angle:.1f}도가 될 때까지 기다립니다."
        )
        deadline = time.monotonic() + self.initial_tracking_timeout
        last_info = None

        while time.monotonic() < deadline:
            _, info = self._observe("비학습 워밍업")
            last_info = info
            if info["angle_detected"] and abs(info["angle_degrees"]) > self.terminal_angle:
                self._log(
                    f"[비학습 워밍업 종료] |각도|="
                    f"{abs(info['angle_degrees']):.2f}도 > "
                    f"{self.terminal_angle:.1f}도"
                )
                self._log(
                    f"[조작 정지] {self.idle_after_terminal:.2f}초 동안 "
                    "아무 키도 누르지 않습니다."
                )
                time.sleep(self.idle_after_terminal)
                self._press_ready_space("학습 에피소드 1 초기화")
                self._log(
                    f"[다음 에피소드 준비] |각도| <= "
                    f"{self.upright_angle:.1f}도를 확인한 뒤 시작 Space를 누릅니다."
                )
                time.sleep(self.idle_after_terminal)
                self._press_ready_space("시작 스페이스 입력")
                self._initial_untrained_game_running = False
                self._angle_history.clear()
                self._previous_angle = 0.0
                return
            time.sleep(0.05)

        raise RuntimeError(
            f"비학습 워밍업 게임이 {self.initial_tracking_timeout:.1f}초 안에 "
            f"|각도| > {self.terminal_angle:.1f} 조건을 만족하지 못했습니다. "
            f"마지막 상태: {last_info}"
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        # SB3 모델 생성이 끝나고 처음 reset()이 호출되는 시점입니다.
        # 이미 진행 중인 첫 게임은 학습에서 버리고 정상 재시작 후 EP 1을 반환합니다.
        if self._initial_untrained_game_running:
            self._finish_initial_untrained_game()

        # 정상 종료라면 step()에서 1초 대기와 준비 Space 입력까지 끝난 상태입니다.
        # 게임 도중 강제 reset 또는 시간 제한 종료만 전체 페이지를 다시 준비합니다.
        if self._episode_active or self._needs_full_reload or options.get("reload", False):
            self._reload_from_beginning()

        if not self._ready_space_pressed:
            raise RuntimeError(
                "준비 Space가 입력되지 않은 상태입니다. 게임 초기화 순서를 확인하세요."
            )

        self._angle_history.clear()
        self._previous_angle = 0.0
        self._last_line = None
        self._episode_steps = 0
        self._episode_active = False
        self._needs_full_reload = False

        observation, info = self._wait_until_upright()

        self._page.keyboard.press("Space")
        self._ready_space_pressed = False
        self._episode_number += 1
        self._episode_active = True
        self._log(
            f"[에피소드 {self._episode_number} 시작] "
            f"직립 각도={info['angle_degrees']:+.2f}도, 시작 Space 입력 완료"
        )
        time.sleep(0.10)

        observation, info = self._observe("진행 중")
        info["episode_number"] = self._episode_number
        info["episode_started"] = True

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if not self._episode_active:
            raise RuntimeError("step() 전에 reset()을 호출해야 합니다.")
        if not self.action_space.contains(action):
            raise ValueError(
                "action은 0(왼쪽), 1(오른쪽), 2(무행동) 중 하나여야 합니다."
            )

        action_number = int(action)
        if action_number == 0:
            key = "ArrowLeft"
            action_name = "LEFT"
        elif action_number == 1:
            key = "ArrowRight"
            action_name = "RIGHT"
        else:
            key = None
            action_name = "NOOP"

        if key is None:
            # 방향키를 누르지 않되 다른 행동과 동일한 시간 동안 기다립니다.
            time.sleep(self.action_duration)
        else:
            self._page.keyboard.down(key)
            try:
                time.sleep(self.action_duration)
            finally:
                # 종료 여부와 관계없이 방향키는 반드시 놓습니다.
                self._page.keyboard.up(key)
        time.sleep(self.observation_delay)

        self._episode_steps += 1
        observation, info = self._observe("진행 중")
        current_angle = info["angle_degrees"]

        terminated = bool(
            info["angle_detected"] and abs(current_angle) > self.terminal_angle
        )
        truncated = self._episode_steps >= self.max_episode_steps

        reward = 1.0 - 0.25 * (abs(current_angle) / self.terminal_angle) ** 2
        termination_reason = None

        if terminated:
            termination_reason = "abs_angle_over_limit"
            reward -= 10.0
            self._episode_active = False
            self._log(
                f"[에피소드 {self._episode_number} 종료] "
                f"|각도|={abs(current_angle):.2f}도 > {self.terminal_angle:.1f}도"
            )
            self._log(
                f"[조작 정지] {self.idle_after_terminal:.2f}초 동안 "
                "어떤 키도 누르지 않습니다."
            )
            time.sleep(self.idle_after_terminal)
            self._press_ready_space(f"에피소드 {self._episode_number} 종료 후 초기화")
            self._log(
                f"[다음 에피소드 준비] reset()에서 |각도| <= "
                f"{self.upright_angle:.1f}도를 확인한 뒤 시작 Space를 누릅니다."
            )
            time.sleep(self.idle_after_terminal)
            self._press_ready_space("시작 스페이스 입력")
        elif truncated:
            termination_reason = "time_limit"
            self._episode_active = False
            self._needs_full_reload = True
            self._log(
                f"[에피소드 {self._episode_number} 시간 제한] "
                f"step={self._episode_steps}"
            )

        info["episode_number"] = self._episode_number
        info["action"] = action_number
        info["action_name"] = action_name
        info["termination_reason"] = termination_reason

        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self._last_frame is None:
            return None
        if self.render_mode == "rgb_array":
            return cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
        if self.render_mode == "human":
            debug = self._last_frame.copy()
            if self._last_line is not None:
                x1, y1, x2, y2 = self._last_line
                cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                debug,
                f"EP {self._episode_number} | angle {self._previous_angle:+.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Stork RL Environment Test", debug)
            cv2.waitKey(1)
        return None

    def close(self):
        self._episode_active = False
        cv2.destroyAllWindows()
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None


if __name__ == "__main__":
    # 현재는 환경 전환 테스트용으로 무작위 좌/우 행동을 수행합니다.
    env = StorkGameEnv(render_mode="human", headless=False, print_status=True)
    try:
        for _ in range(3):
            observation, info = env.reset()
            total_reward = 0.0

            while True:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    print(
                        f"[테스트 결과] episode={info['episode_number']} | "
                        f"steps={info['episode_steps']} | "
                        f"reward={total_reward:.2f} | "
                        f"reason={info['termination_reason']}",
                        flush=True,
                    )
                    break
    finally:
        env.close()
