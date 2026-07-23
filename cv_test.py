import math
import time
from collections import deque

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


GAME_URL = "https://vidkidz.tistory.com/2825"

# 새가 화면 중앙 근처에 있다는 가정으로 탐색 영역을 제한합니다.
# 누운 새까지 포함하도록 화면 아래쪽도 넓게 탐색합니다.
# 값의 순서는 화면 너비/높이에 대한 (왼쪽, 위, 오른쪽, 아래) 비율입니다.
BIRD_ROI = (0.20, 0.15, 0.90, 0.96)


def detect_neck_angle(frame):
    """
    threshold 영상에서 가장 긴 목 직선을 찾아 수직선과의 각도를 계산합니다.

    반환값
      angle: 수직 기준 각도. 오른쪽으로 기울면 +, 왼쪽이면 - (도 단위)
      line:   검출된 목 직선의 (x1, y1, x2, y2)
      thresh: 화면 표시용 임계 영상
    """
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 기존과 같은 임계 영상: 검은 선=0, 밝은 배경=255
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # 검은 선을 흰색으로 반전한 뒤 작은 끊김을 연결합니다.
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
        theta=np.pi / 360,  # 0.5도 간격
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

        # 직선은 방향이 없으므로 같은 선이 항상 -90~+90도에 오도록 정규화합니다.
        raw_angle = math.degrees(math.atan2(dx, -dy))
        angle = (raw_angle + 90) % 180 - 90

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # 목은 보통 새의 중앙에 있으므로 화면 가장자리의 긴 선보다 우선합니다.
        center_distance = math.hypot(
            (mid_x - width * 0.53) * 0.35,
            (mid_y - height * 0.60) * 0.20,
        )
        score = length - center_distance
        candidates.append((score, length, angle, (x1, y1, x2, y2)))

    if not candidates:
        return None, None, thresh, None

    # 다리와 부리보다 긴 목 선이 가장 높은 점수를 받습니다.
    _, _, angle, line = max(candidates, key=lambda item: item[0])
    return angle, line, thresh, None


with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--window-size=960,540"],
    )

    page = browser.new_page(viewport={"width": 960, "height": 540})
    page.goto(GAME_URL)
    time.sleep(20)

    page.mouse.click(480, 270)
    print("중앙 클릭 완료")
    time.sleep(1)

    page.keyboard.press("Space")
    print("게임 시작할 준비 완료")
    time.sleep(2)

    # 최근 각도들의 중앙값을 사용해 화면 떨림을 줄입니다.
    angle_history = deque(maxlen=5)

    try:
        while True:
            screenshot = page.screenshot()
            image = np.frombuffer(screenshot, dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            angle, neck_line, thresh, shapes = detect_neck_angle(frame)

            # 흑백 threshold 영상을 컬러로 바꾸면 검출선과 글자를 색으로 그릴 수 있습니다.
            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            height, width = frame.shape[:2]
            left = int(width * BIRD_ROI[0])
            top = int(height * BIRD_ROI[1])
            right = int(width * BIRD_ROI[2])
            bottom = int(height * BIRD_ROI[3])
            cv2.rectangle(debug, (left, top), (right, bottom), (255, 0, 0), 1)

            if angle is not None:
                # 검출된 직선에는 앞뒤 방향이 없으므로 180도 차이의 같은 각도 중
                # 이전 프레임과 더 자연스럽게 이어지는 값을 선택합니다.
                if angle_history:
                    while angle - angle_history[-1] > 90:
                        angle -= 180
                    while angle - angle_history[-1] < -90:
                        angle += 180
                angle_history.append(angle)
                smooth_angle = float(np.median(angle_history))
                smooth_angle = (smooth_angle + 180) % 360 - 180

                x1, y1, x2, y2 = neck_line
                cv2.line(
                    debug,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3,
                )
                text = f"Neck: {smooth_angle:+.1f} deg"
                color = (0, 0, 255)
                print(f"\r목 각도: {smooth_angle:+6.1f}도", end="", flush=True)
            else:
                angle_history.clear()
                text = "Neck: not found"
                color = (0, 0, 255)

            cv2.putText(
                debug,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Threshold + Neck Angle", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        browser.close()
