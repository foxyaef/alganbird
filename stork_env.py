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
    행동: 0=왼쪽 방향키, 1=오른쪽 방향키
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
        idle_after_terminal=5.0,
        upright_timeout=15.0,
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
        self.action_duration = float(action_duration)
        self.observation_delay = float(observation_delay)
        self.max_episode_steps = int(max_episode_steps)
        self.print_status = bool(print_status)

        self.action_space = spaces.Discrete(2)
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
            self._log("[초기화] 화면 중앙 클릭 완료")
            time.sleep(1.0)
            self._press_ready_space("에피소드 1 준비")
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
        self._page.keyboard.press("Space")
        
        

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
            "episode": display_episode,
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
        while time.monotonic() < deadline:
            observation, info = self._observe("직립 대기")
            last_info = info
            if info["angle_detected"] and abs(info["angle_degrees"]) <= self.upright_angle:
                self._log(
                    f"[직립 확인] 현재 각도 {info['angle_degrees']:+.2f}도 "
                    f"-> 시작 Space를 누릅니다."
                )
                return observation, info
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

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
        info["episode"] = self._episode_number
        info["episode_started"] = True

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if not self._episode_active:
            raise RuntimeError("step() 전에 reset()을 호출해야 합니다.")
        if not self.action_space.contains(action):
            raise ValueError("action은 0(왼쪽) 또는 1(오른쪽)이어야 합니다.")

        key = "ArrowLeft" if int(action) == 0 else "ArrowRight"
        action_name = "LEFT" if int(action) == 0 else "RIGHT"
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
            self._press_ready_space(f"에피소드 {self._episode_number} 종료 후 초기화")
        elif truncated:
            termination_reason = "time_limit"
            self._episode_active = False
            self._needs_full_reload = True
            self._log(
                f"[에피소드 {self._episode_number} 시간 제한] "
                f"step={self._episode_steps}"
            )

        info["episode"] = self._episode_number
        info["action"] = int(action)
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
                        f"[테스트 결과] episode={info['episode']} | "
                        f"steps={info['episode_steps']} | "
                        f"reward={total_reward:.2f} | "
                        f"reason={info['termination_reason']}",
                        flush=True,
                    )
                    break
    finally:
        env.close()
