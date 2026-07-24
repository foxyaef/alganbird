import time
import traceback

import cv2
import numpy as np

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


GAME_URL = "https://vidkidz.tistory.com/2825"


with sync_playwright() as p:

    # 브라우저 실행
    browser = p.chromium.launch(
        headless=False,
        args=[
            "--window-size=960,540"
        ]
    )

    # 새 페이지 생성
    page = browser.new_page(
        viewport={
            "width": 960,
            "height": 540
        }
    )

    try:
        # 게임 사이트 접속
        try:
            page.goto(
                GAME_URL,
                wait_until="domcontentloaded",
                timeout=60_000
            )
            print("사이트 접속 완료")

        except PlaywrightTimeoutError:
            print("사이트 로딩 시간이 초과됐지만 계속 진행합니다.")

        time.sleep(5)

        # 페이지 중앙 클릭
        print("중앙 클릭 시도")
        page.mouse.click(480, 270)
        print("중앙 클릭 완료")

        time.sleep(2)

        # 게임 시작
        page.keyboard.press("Space")
        print("게임 시작 키 입력 완료")

        time.sleep(2)

        print("영상 분석 시작")
        print("OpenCV 창에서 q를 누르면 종료됩니다.")

        while True:

            # 페이지 스크린샷 캡처
            screenshot = page.screenshot()

            # bytes를 NumPy 배열로 변환
            image = np.frombuffer(
                screenshot,
                dtype=np.uint8
            )

            # NumPy 배열을 OpenCV 이미지로 변환
            frame = cv2.imdecode(
                image,
                cv2.IMREAD_COLOR
            )

            # 이미지 변환 실패 여부 확인
            if frame is None:
                print("스크린샷 변환에 실패했습니다.")
                continue

            # 컬러 이미지를 흑백으로 변환
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # 흑백 이미지 이진화 및 색상 반전
            _, thresh = cv2.threshold(
                gray,
                120,
                255,
                cv2.THRESH_BINARY_INV
            )

            # 탐색할 ROI 영역 좌표
            x1 = 200
            y1 = 200

            x2 = 750
            y2 = 500

            # 이진화 이미지에서 ROI 영역 잘라내기
            roi_thresh = thresh[y1:y2, x1:x2]

            # ROI 영역에서 윤곽선 찾기
            contours, _ = cv2.findContours(
                roi_thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 윤곽선이 존재하는 경우
            if len(contours) > 0:

                # 가장 큰 윤곽선을 황새로 간주
                largest_contour = max(
                    contours,
                    key=cv2.contourArea
                )

                # 윤곽선 내부를 채우기 위한 빈 마스크
                mask = np.zeros_like(roi_thresh)

                # 가장 큰 윤곽선 내부를 흰색으로 채우기
                cv2.drawContours(
                    mask,
                    [largest_contour],
                    -1,
                    255,
                    thickness=cv2.FILLED
                )

                # 채워진 마스크에서 윤곽선 다시 찾기
                filled_contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # 채워진 윤곽선이 존재하는 경우
                if len(filled_contours) > 0:

                    largest_contour = max(
                        filled_contours,
                        key=cv2.contourArea
                    )

                    # 윤곽선 넓이 계산
                    area = cv2.contourArea(
                        largest_contour
                    )

                    # 너무 작은 물체는 무시
                    if area > 30:

                        # ROI 좌표를 전체 이미지 좌표로 보정
                        shifted_contour = (
                            largest_contour
                            + np.array([x1, y1])
                        )

                        # 감지된 물체를 초록색으로 표시
                        cv2.drawContours(
                            frame,
                            [shifted_contour],
                            -1,
                            (0, 255, 0),
                            thickness=cv2.FILLED
                        )

            # ROI 영역을 파란색 사각형으로 표시
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # OpenCV 창 출력
            cv2.imshow("Game Capture", frame)
            cv2.imshow("Gray", gray)
            cv2.imshow("Threshold", thresh)

            # q 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("q 입력: 프로그램을 종료합니다.")
                break

            time.sleep(0.01)

    except Exception:
        # 오류 내용을 터미널에 자세히 출력
        print("\n프로그램 실행 중 오류가 발생했습니다.")
        traceback.print_exc()

        # 오류를 읽기 전에 창이 닫히는 것을 방지
        input("\n오류를 확인한 후 Enter를 누르세요.")

    finally:
        # OpenCV 창 종료
        cv2.destroyAllWindows()

        # 브라우저 종료
        browser.close()

        print("프로그램 종료 완료")