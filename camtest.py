from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os

app = Flask(__name__)
socketio = SocketIO(app)

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@socketio.on('connect')
def handle_connect():
    print("클라이언트가 연결되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    print("클라이언트와의 연결이 종료되었습니다.")

@socketio.on('frame')
def handle_frame(data):
    try:
        # 클라이언트로부터 받은 데이터 처리
        frame_data = data['frame']
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ROI 및 원근 변환 처리
        roi_frame, src, top_view,cropped_top_view = process_roi_and_perspective(frame)

        # 이미지 저장
        save_debug_images(frame, roi_frame, src, top_view, cropped_top_view )

        # 디스플레이는 선택사항
        # cv2.imshow("Top View", top_view)
        # cv2.imshow("Original with src", frame_with_src)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     socketio.stop()

    except Exception as e:
        print(f"프레임 처리 중 오류 발생: {e}")

def process_roi_and_perspective(frame):
    """ROI 처리 및 원근 변환."""
    height, width = frame.shape[:2]

    # ROI 설정
    lower_bound = int(height * 0.35)
    upper_bound = int(height * 0.5)

    # ROI 범위 마스크
    mask = np.zeros_like(frame, dtype=np.uint8)
    roi_vertices = np.array([[
        (0, height),               # Bottom-left
        (0, lower_bound),          # Left-middle
        (width, lower_bound),      # Right-middle
        (width, height)            # Bottom-right
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)

    #src 좌표 설정 (ROI 기준으로 정의)**
    src = np.float32([
        [width * 0.35, lower_bound],  # Top-left
        [width * 0.7, lower_bound],  # Top-right
        [width * 1.0, height],       # Bottom-right
        [width * 0.0, height]        # Bottom-left
    ])

    #dst 좌표 설정 (Top View 기준)**
    dst = np.float32([
        [width * 0.3, 0],              # Top-left
        [width * 0.75, 0],              # Top-right
        [width * 1.0, height],         # Bottom-right
        [width * 0.0, height]          # Bottom-left
    ])


    # 원근 변환 수행
    M = cv2.getPerspectiveTransform(src, dst)
    top_view = cv2.warpPerspective(frame, M, (width, height), flags=cv2.INTER_LINEAR)
     # 하단의 불필요한 부분을 잘라내기 (cutoff 적용)
    cutoff = int(height * 0.85)  # 하단 15%를 제거
    cropped_top_view = top_view[:cutoff, :]  # 상단부터 cutoff까지만 유지


    return roi_frame, src, top_view,cropped_top_view

def save_debug_images(frame, roi_frame, src, top_view,cropped_top_view):
    """디버깅용 이미지를 저장."""
    car_num = "debug"  # 차량 ID 또는 고유 식별자를 사용할 수 있음

    # 원본 이미지 저장
    cv2.imwrite(f"{OUTPUT_DIR}/original.jpg", frame)

    # ROI 처리된 이미지 저장
    cv2.imwrite(f"{OUTPUT_DIR}/_roi.jpg", roi_frame)

    # Top View 저장
    cv2.imwrite(f"{OUTPUT_DIR}/_top_view.jpg", top_view)
    cv2.imwrite(f"{OUTPUT_DIR}/cropped_top_view.jpg", cropped_top_view)

    # src 좌표 시각화 및 저장
    frame_with_src = frame.copy()
    for point in src:
        cv2.circle(frame_with_src, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # src는 빨간 점으로 표시
    cv2.imwrite(f"src_visualization.jpg", frame_with_src)

    print(f"이미지가 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
