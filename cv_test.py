import time
import math

import cv2
import numpy as np

from playwright.sync_api import sync_playwright


GAME_URL = "https://vidkidz.tistory.com/2825"


def normalize_vector(vector):
    """
    2차원 벡터를 단위 벡터로 변환한다.
    """
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    length = np.linalg.norm(vector)

    if length < 1e-6:
        return None

    return vector / length


def angle_between_axes(vector1, vector2):
    """
    두 축 사이의 작은 각도를 계산한다.

    축은 방향성이 없기 때문에
    v와 -v를 같은 축으로 취급한다.

    반환 범위:
        0도 ~ 90도
    """
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)

    if vector1 is None or vector2 is None:
        return None

    cosine = abs(
        np.dot(vector1, vector2)
    )

    cosine = np.clip(
        cosine,
        0.0,
        1.0
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    return angle


def signed_angle_from_vertical(vector):
    """
    목 벡터가 수직선에서 좌우로 얼마나 기울어졌는지 계산한다.

    화면 좌표계:
        x축: 오른쪽이 양수
        y축: 아래쪽이 양수

    반환값:
        음수: 왼쪽 기울기
        양수: 오른쪽 기울기
        0도: 수직
    """
    vector = normalize_vector(vector)

    if vector is None:
        return None

    # 방향을 아래쪽으로 통일
    if vector[1] < 0:
        vector = -vector

    angle = math.degrees(
        math.atan2(
            vector[0],
            vector[1]
        )
    )

    return angle


def find_bird_bbox(black_mask):
    """
    검은색 외곽선으로부터 새의 대략적인 영역을 찾는다.
    """
    contours, _ = cv2.findContours(
        black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_height, image_width = black_mask.shape

    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # 너무 작은 점이나 잡음 제거
        if width < 15 or height < 25:
            continue

        center_x = x + width / 2
        center_y = y + height / 2

        distance_from_center = (
            abs(center_x - image_width / 2)
            + abs(center_y - image_height / 2)
        )

        area_score = width * height
        center_score = distance_from_center * 5

        score = area_score - center_score

        candidates.append(
            (
                score,
                (x, y, width, height)
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0][1]


def detect_head(gray, bird_bbox):
    """
    Hough Circle Transform으로 새의 머리를 찾는다.
    """
    x, y, width, height = bird_bbox

    # 새의 위쪽 부분만 머리 후보 영역으로 사용
    head_area_height = max(
        1,
        int(height * 0.50)
    )

    roi = gray[
        y:y + head_area_height,
        x:x + width
    ]

    if roi.size == 0:
        return None

    blurred = cv2.GaussianBlur(
        roi,
        (7, 7),
        1.5
    )

    min_radius = max(
        8,
        int(height * 0.035)
    )

    max_radius = max(
        min_radius + 2,
        int(height * 0.15)
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.round(
        circles[0]
    ).astype(int)

    # 머리는 일반적으로 새 영역의 위쪽 중앙 근처
    expected_x = width * 0.55
    expected_y = height * 0.12

    best_circle = min(
        circles,
        key=lambda circle: (
            abs(circle[0] - expected_x)
            + abs(circle[1] - expected_y)
        )
    )

    head_x = x + best_circle[0]
    head_y = y + best_circle[1]
    radius = best_circle[2]

    return head_x, head_y, radius


def fit_neck_axis(black_mask, head):
    """
    머리 아래쪽의 검은 선을 사용해 목 방향을 추정한다.
    """
    head_x, head_y, radius = head

    image_height, image_width = black_mask.shape

    x_margin = max(
        10,
        int(radius * 0.65)
    )

    x1 = max(
        0,
        head_x - x_margin
    )

    x2 = min(
        image_width,
        head_x + x_margin
    )

    y1 = max(
        0,
        head_y + int(radius * 0.55)
    )

    y2 = min(
        image_height,
        head_y + int(radius * 5.5)
    )

    neck_roi = black_mask[
        y1:y2,
        x1:x2
    ]

    if neck_roi.size == 0:
        return None

    ys, xs = np.where(
        neck_roi > 0
    )

    if len(xs) < 20:
        return None

    global_xs = xs + x1
    global_ys = ys + y1

    points = np.column_stack(
        (
            global_xs,
            global_ys
        )
    ).astype(np.float32)

    line_result = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    vx, vy, line_x, line_y = (
        line_result.flatten()
    )

    # 목 방향을 아래쪽으로 통일
    if vy < 0:
        vx = -vx
        vy = -vy

    return {
        "vector": np.array(
            [vx, vy],
            dtype=np.float32
        ),
        "point": np.array(
            [line_x, line_y],
            dtype=np.float32
        ),
        "roi": (
            x1,
            y1,
            x2,
            y2
        )
    }


def fit_body_axis(black_mask, head):
    """
    몸통 영역의 검은 픽셀에 PCA를 적용하여
    몸통의 주축을 추정한다.
    """
    head_x, head_y, radius = head

    image_height, image_width = black_mask.shape

    x1 = max(
        0,
        head_x - int(radius * 5.0)
    )

    x2 = min(
        image_width,
        head_x + int(radius * 1.5)
    )

    y1 = max(
        0,
        head_y + int(radius * 3.5)
    )

    y2 = min(
        image_height,
        head_y + int(radius * 7.8)
    )

    body_roi = black_mask[
        y1:y2,
        x1:x2
    ]

    if body_roi.size == 0:
        return None

    ys, xs = np.where(
        body_roi > 0
    )

    if len(xs) < 30:
        return None

    global_xs = xs + x1
    global_ys = ys + y1

    points = np.column_stack(
        (
            global_xs,
            global_ys
        )
    ).astype(np.float32)

    center = np.mean(
        points,
        axis=0
    )

    centered_points = points - center

    covariance = np.cov(
        centered_points,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    main_index = np.argmax(
        eigenvalues
    )

    body_vector = eigenvectors[
        :,
        main_index
    ]

    # 몸통 방향을 오른쪽으로 통일
    if body_vector[0] < 0:
        body_vector = -body_vector

    return {
        "vector": body_vector,
        "center": center,
        "roi": (
            x1,
            y1,
            x2,
            y2
        )
    }


def calculate_bird_angle(frame):
    """
    프레임에서 새의 머리, 목, 몸통을 검출하고
    목과 몸통 사이 각도를 계산한다.

    반환값:
        neck_body_angle
        neck_tilt
        debug_frame
        black_mask
    """
    debug_frame = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # 검은 외곽선 추출
    black_mask = cv2.inRange(
        gray,
        0,
        80
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    bird_bbox = find_bird_bbox(
        black_mask
    )

    if bird_bbox is None:
        cv2.putText(
            debug_frame,
            "Bird not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return (
            None,
            None,
            debug_frame,
            black_mask
        )

    x, y, width, height = bird_bbox

    cv2.rectangle(
        debug_frame,
        (x, y),
        (
            x + width,
            y + height
        ),
        (255, 0, 0),
        2
    )

    head = detect_head(
        gray,
        bird_bbox
    )

    if head is None:
        cv2.putText(
            debug_frame,
            "Head not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return (
            None,
            None,
            debug_frame,
            black_mask
        )

    head_x, head_y, radius = head

    cv2.circle(
        debug_frame,
        (head_x, head_y),
        radius,
        (0, 255, 0),
        2
    )

    cv2.circle(
        debug_frame,
        (head_x, head_y),
        4,
        (0, 0, 255),
        -1
    )

    neck = fit_neck_axis(
        black_mask,
        head
    )

    body = fit_body_axis(
        black_mask,
        head
    )

    if neck is None:
        cv2.putText(
            debug_frame,
            "Neck not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return (
            None,
            None,
            debug_frame,
            black_mask
        )

    if body is None:
        cv2.putText(
            debug_frame,
            "Body not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return (
            None,
            None,
            debug_frame,
            black_mask
        )

    neck_vector = neck["vector"]
    neck_point = neck["point"]

    body_vector = body["vector"]
    body_center = body["center"]

    neck_body_angle = angle_between_axes(
        neck_vector,
        body_vector
    )

    neck_tilt = signed_angle_from_vertical(
        neck_vector
    )

    line_length = max(
        80,
        int(radius * 4)
    )

    neck_start = (
        int(
            neck_point[0]
            - neck_vector[0] * line_length
        ),
        int(
            neck_point[1]
            - neck_vector[1] * line_length
        )
    )

    neck_end = (
        int(
            neck_point[0]
            + neck_vector[0] * line_length
        ),
        int(
            neck_point[1]
            + neck_vector[1] * line_length
        )
    )

    cv2.line(
        debug_frame,
        neck_start,
        neck_end,
        (0, 0, 255),
        3
    )

    body_start = (
        int(
            body_center[0]
            - body_vector[0] * line_length
        ),
        int(
            body_center[1]
            - body_vector[1] * line_length
        )
    )

    body_end = (
        int(
            body_center[0]
            + body_vector[0] * line_length
        ),
        int(
            body_center[1]
            + body_vector[1] * line_length
        )
    )

    cv2.line(
        debug_frame,
        body_start,
        body_end,
        (255, 0, 255),
        3
    )

    cv2.circle(
        debug_frame,
        tuple(
            body_center.astype(int)
        ),
        5,
        (255, 255, 0),
        -1
    )

    neck_roi = neck["roi"]
    body_roi = body["roi"]

    cv2.rectangle(
        debug_frame,
        (
            neck_roi[0],
            neck_roi[1]
        ),
        (
            neck_roi[2],
            neck_roi[3]
        ),
        (0, 255, 255),
        1
    )

    cv2.rectangle(
        debug_frame,
        (
            body_roi[0],
            body_roi[1]
        ),
        (
            body_roi[2],
            body_roi[3]
        ),
        (255, 255, 0),
        1
    )

    cv2.putText(
        debug_frame,
        f"Neck-Body: {neck_body_angle:.1f} deg",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        debug_frame,
        f"Neck tilt: {neck_tilt:+.1f} deg",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        debug_frame,
        "Red: neck axis",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        debug_frame,
        "Magenta: body axis",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    return (
        neck_body_angle,
        neck_tilt,
        debug_frame,
        black_mask
    )


def main():
    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False,
            args=[
                "--window-size=960,540"
            ]
        )

        page = browser.new_page(
            viewport={
                "width": 960,
                "height": 540
            }
        )

        try:
            print("게임 사이트 접속 중...")

            page.goto(
                GAME_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            time.sleep(20)

            page.mouse.click(
                480,
                270
            )

            print("중앙 클릭 완료")

            time.sleep(1)

            page.keyboard.press(
                "Space"
            )

            print("게임 시작")
            print("종료하려면 OpenCV 창에서 q를 누르세요.")

            time.sleep(2)

            while True:

                screenshot = page.screenshot()

                image = np.frombuffer(
                    screenshot,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    image,
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    print(
                        "\n스크린샷 변환에 실패했습니다."
                    )
                    continue

                (
                    neck_body_angle,
                    neck_tilt,
                    angle_frame,
                    black_mask
                ) = calculate_bird_angle(
                    frame
                )

                if neck_body_angle is not None:
                    print(
                        "\r"
                        f"목-몸통 각도: "
                        f"{neck_body_angle:6.2f}도 | "
                        f"목 기울기: "
                        f"{neck_tilt:+6.2f}도",
                        end="",
                        flush=True
                    )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                _, threshold = cv2.threshold(
                    gray,
                    120,
                    255,
                    cv2.THRESH_BINARY
                )

                cv2.imshow(
                    "Game Capture",
                    frame
                )

                cv2.imshow(
                    "Bird Angle Detection",
                    angle_frame
                )

                cv2.imshow(
                    "Black Line Mask",
                    black_mask
                )

                cv2.imshow(
                    "Threshold",
                    threshold
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\n종료합니다.")
                    break

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n키보드 입력으로 종료합니다.")

        except Exception as error:
            print(
                f"\n오류가 발생했습니다: {error}"
            )

        finally:
            cv2.destroyAllWindows()
            browser.close()


if __name__ == "__main__":
    main()