import time
import cv2
import numpy as np

from playwright.sync_api import sync_playwright


GAME_URL = "https://vidkidz.tistory.com/2825"

with sync_playwright() as p:

    # 브라우저 실행
    browser = p.chromium.launch(
        headless=False,
        args=[
            "--window-size=960,540"
        ]
    )

    # 페이지 생성
    page = browser.new_page(
        viewport={
            "width": 960,
            "height": 540
        }
    )

    # 게임 사이트 접속
    page.goto(GAME_URL)
    time.sleep(3)

    # 중앙 클릭
    page.mouse.click(480, 270)
    print("중앙 클릭 완료")
    time.sleep(1)

    # 게임 시작
    page.keyboard.press("Space")
    print("게임 시작할 준비 완")
    time.sleep(2)
    

    # input("엔터를 쳐서 종료하세요")를 드래그 하고 붙여넣어 주세요.
    # playwright 안에 들어가도록 들여쓰기 해야해요.
    while True:

        # 페이지 스크린샷 캡처
        screenshot = page.screenshot()

        # bytes -> numpy array
        image = np.frombuffer(
            screenshot,
            dtype=np.uint8
        )

        # numpy -> OpenCV image
        frame = cv2.imdecode(
            image,
            cv2.IMREAD_COLOR
        )
        
        # 원본 출력
        # cv2.imshow("Game Capture", frame)
        # 위의 코드 드래그 후 해당 내용 복붙

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # cv2.imshow("Gray", gray)
# 위의 코드 드래그 후 해당 내용 복붙

        _, thresh = cv2.threshold(
            gray,
            120,
            255,
            cv2.THRESH_BINARY
        )

        cv2.imshow("Threshold", thresh)
                        
        time.sleep(0.01)
                
        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break