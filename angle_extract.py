import math
import time
from collections import deque

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


GAME_URL = "https://vidkidz.tistory.com/2825"

# ?덇? ?붾㈃ 以묒븰 洹쇱쿂???덈떎??媛?뺤쑝濡??먯깋 ?곸뿭???쒗븳?⑸땲??
# ?꾩슫 ?덇퉴吏 ?ы븿?섎룄濡??붾㈃ ?꾨옒履쎈룄 ?볤쾶 ?먯깋?⑸땲??
# 媛믪쓽 ?쒖꽌???붾㈃ ?덈퉬/?믪씠?????(?쇱そ, ?? ?ㅻⅨ履? ?꾨옒) 鍮꾩쑉?낅땲??
BIRD_ROI = (0.20, 0.15, 0.90, 0.96)


def detect_neck_angle(frame):
    """
    threshold ?곸긽?먯꽌 媛??湲?紐?吏곸꽑??李얠븘 ?섏쭅?좉낵??媛곷룄瑜?怨꾩궛?⑸땲??

    諛섑솚媛?
      angle: ?섏쭅 湲곗? 媛곷룄. ?ㅻⅨ履쎌쑝濡?湲곗슱硫?+, ?쇱そ?대㈃ - (???⑥쐞)
      line:   寃異쒕맂 紐?吏곸꽑??(x1, y1, x2, y2)
      thresh: ?붾㈃ ?쒖떆???꾧퀎 ?곸긽
    """
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 湲곗〈怨?媛숈? ?꾧퀎 ?곸긽: 寃? ??0, 諛앹? 諛곌꼍=255
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # 寃? ?좎쓣 ?곗깋?쇰줈 諛섏쟾?????묒? ?딄????곌껐?⑸땲??
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
        theta=np.pi / 360,  # 0.5??媛꾧꺽
        threshold=max(20, int(height * 0.025)),
        minLineLength=max(35, int(height * 0.08)),
        maxLineGap=max(12, int(height * 0.03)),
    )

    if lines is None:
        return None, None, thresh, None

    candidates = []
    for roi_x1, roi_y1, roi_x2, roi_y2 in lines.reshape(-1, 4):
        x1, y1 = int(roi_x1 + left), int(roi_y1 + top)
        x2, y2 = int(roi_x2 + left), int(roi_y2 + top)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        # 吏곸꽑? 諛⑺뼢???놁쑝誘濡?媛숈? ?좎씠 ??긽 -90~+90?꾩뿉 ?ㅻ룄濡??뺢퇋?뷀빀?덈떎.
        raw_angle = math.degrees(math.atan2(dx, -dy))
        angle = (raw_angle + 90) % 180 - 90

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # 紐⑹? 蹂댄넻 ?덉쓽 以묒븰???덉쑝誘濡??붾㈃ 媛?μ옄由ъ쓽 湲??좊낫???곗꽑?⑸땲??
        center_distance = math.hypot(
            (mid_x - width * 0.53) * 0.35,
            (mid_y - height * 0.60) * 0.20,
        )
        score = length - center_distance
        candidates.append((score, length, angle, (x1, y1, x2, y2)))

    if not candidates:
        return None, None, thresh, None

    # ?ㅻ━? 遺由щ낫??湲?紐??좎씠 媛???믪? ?먯닔瑜?諛쏆뒿?덈떎.
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
    print("center clicked")
    time.sleep(1)

    page.keyboard.press("Space")
    print("game started")
    time.sleep(2)

    # 理쒓렐 媛곷룄?ㅼ쓽 以묒븰媛믪쓣 ?ъ슜???붾㈃ ?⑤┝??以꾩엯?덈떎.
    angle_history = deque(maxlen=5)

    try:
        while True:
            screenshot = page.screenshot()
            image = np.frombuffer(screenshot, dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            angle, neck_line, thresh, shapes = detect_neck_angle(frame)

            # ?묐갚 threshold ?곸긽??而щ윭濡?諛붽씀硫?寃異쒖꽑怨?湲?먮? ?됱쑝濡?洹몃┫ ???덉뒿?덈떎.
            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            height, width = frame.shape[:2]
            left = int(width * BIRD_ROI[0])
            top = int(height * BIRD_ROI[1])
            right = int(width * BIRD_ROI[2])
            bottom = int(height * BIRD_ROI[3])
            cv2.rectangle(debug, (left, top), (right, bottom), (255, 0, 0), 1)

            if angle is not None:
                # 寃異쒕맂 吏곸꽑?먮뒗 ?욌뮘 諛⑺뼢???놁쑝誘濡?180??李⑥씠??媛숈? 媛곷룄 以?
                # ?댁쟾 ?꾨젅?꾧낵 ???먯뿰?ㅻ읇寃??댁뼱吏??媛믪쓣 ?좏깮?⑸땲??
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
                # Angle is shown in the OpenCV window.
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


