from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import time
from collections import deque
import threading
import torch

app = Flask(__name__)
socketio = SocketIO(app)

# 저장 디렉토리 설정
OUTPUT_DIR = "frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PID 제어기 초기화
previous_error = 0
integral = 0
kp = 0.3  # 비례 게인 (값을 크게 늘림)
ki = 0.002  # 적분 게인 (값을 유지)
kd = 0.2  # 미분 게인 (값을 크게 늘림)
previous_time = time.time()

# 이동 평균을 위한 큐 설정
lane_center_queue = deque(maxlen=3)
steering_angle_queue = deque(maxlen=3)  # 조향 각도 평균을 위한 큐 설정
previous_lane_center = None  # 이전 차선 중앙값 저장

# 자동차 정보를 저장할 딕셔너리: {sid: 자동차 번호}
cars = {}
car_number = 1
lock = threading.Lock()

# YOLOv5 모델 로드
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    print("YOLOv5 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    model = None
    print(f"모델 로드 중 오류 발생: {e}")
    
@socketio.on('connect')
def handle_connect():
    global car_number
    sid = request.sid
    with lock:
        if sid in cars:
            emit('registration_failed', {'message': '이미 등록된 자동차입니다.'})
            return
        cars[sid] = car_number
        car_number += 1
    emit('registration_success', {'car_number': cars[sid]})
    print(f"자동차 {cars[sid]}이(가) 등록되었습니다.")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    with lock:
        if sid in cars:
            car_num = cars.pop(sid)
            print(f"자동차 {car_num}이(가) 연결을 종료했습니다.")

@socketio.on('frame')
def handle_frame(data):
    global previous_error, integral, previous_time, kp, kd, previous_lane_center
    sid = request.sid
    if sid not in cars:
        emit('error', {'message': '자동차가 등록되지 않았습니다.'})
        return
    car_num = cars[sid]
    try:
        # 클라이언트로부터 받은 데이터 처리
        frame_data = data['frame']
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ROI 및 원근 변환 처리
        roi_frame, src, top_view, cropped_top_view = process_roi_and_perspective(frame)

        # 슬라이딩 윈도우 적용 및 시각화
        sliding_window_result, lane_center = apply_sliding_window(cropped_top_view)

        # 이동 평균을 사용하여 lane_center 값 안정화
        if lane_center is None and previous_lane_center is not None:
            # 차선이 감지되지 않을 경우 이전 차선 위치 사용
            lane_center = previous_lane_center
        elif lane_center is not None:
            previous_lane_center = lane_center

        lane_center_queue.append(lane_center)
        smoothed_lane_center = np.mean(lane_center_queue)

        # 차량 조향 각도 계산 (PID 제어기 사용)
        image_center = cropped_top_view.shape[1] / 2
        error = smoothed_lane_center - image_center
        current_time = time.time()
        delta_time = current_time - previous_time
        delta_time = max(delta_time, 0.01)  # delta_time이 너무 작지 않도록 제한
        previous_time = current_time

        # 곡선 구간에서의 조향 각도 조정을 위해 곡률을 사용한 동적 파라미터 조정
        if abs(error) > 55:  # 작은 오차도 곡선으로 간주하여 빠르게 반응
            kp = 0.45  # 비례 게인을 더 크게 설정하여 빠르게 반응
            kd = 0.3  # 미분 게인을 늘려 급격한 변화 억제
        else:
            kp = 0.3  # 직선 구간에서는 반응을 줄이기 위해 비례 게인을 원래대로
            kd = 0.2  # 미분 게인도 원래대로

        # PID 계산
        proportional = kp * error
        integral += ki * error * delta_time
        derivative = kd * (error - previous_error) / delta_time
        previous_error = error

        steering_angle = proportional + integral + derivative

        # 조향 각도 변환 (30도: 오른쪽, 90도: 중앙, 140도: 왼쪽)
        steering_angle_deg = np.clip(90 + steering_angle, 35, 135)

        # 조향 각도 큐에 추가 및 평균 계산
        steering_angle_queue.append(steering_angle_deg)

        # 이전 각도를 기반으로 현재 조향 각도를 부드럽게 유지
        smoothed_steering_angle_deg = np.mean(steering_angle_queue)

        # 이전 각도와의 큰 차이를 감지하여 부드러운 회전 보장
        if len(steering_angle_queue) > 1:
            previous_angle = steering_angle_queue[-2]
            delta_angle = smoothed_steering_angle_deg - previous_angle
            max_delta = 25  # 각도 변화 허용 범위 설정
            if abs(delta_angle) > max_delta:
                # 각도 변화가 너무 급격할 경우 보정하여 부드러운 회전 유도
                smoothed_steering_angle_deg = previous_angle + np.sign(delta_angle) * max_delta

        # 조향 각도 출력
        print(f"조향 각도 (PID 제어 결과): {smoothed_steering_angle_deg}도")

        # 최종 위치 결정 디버깅 시각화
        arrow_length = 100
        arrow_start = (int(image_center), cropped_top_view.shape[0])
        arrow_end = (
            int(image_center - arrow_length * np.sin(np.radians(smoothed_steering_angle_deg - 90))),
            int(cropped_top_view.shape[0] - arrow_length * np.cos(np.radians(smoothed_steering_angle_deg - 90)))
        )
        cv2.arrowedLine(sliding_window_result, arrow_start, arrow_end, (0, 0, 255), 5)
        cv2.putText(sliding_window_result, f"Steering: {int(smoothed_steering_angle_deg)} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # YOLO 검출 적용
        height = frame.shape[0]
        cropped_frame = frame[height // 2:, :]  # 이미지의 하단 절반을 남김
        results = model(cropped_frame)
        yolo_frame = np.squeeze(results.render())
        cv2.imwrite(f'frames/{car_num}_yolo_frame.jpg', yolo_frame)

        # 사람 검출 시 정지 신호 전송
        should_stop = False
        if results.xyxy[0].size(0) > 0:
            for *box, conf, cls in results.xyxy[0]:
                if int(cls) == 0:  # 클래스 0은 사람입니다.
                    emit('stop_signal', {'reason': 'person_detected'})
                    should_stop = True
                    break
        #print(f"should_stop 상태: {should_stop}, distance: {data.get('distance', None)}, YOLO 감지 개수: {results.xyxy[0].size(0)}")

        # 초음파 센서 거리 확인
        distance = data.get('distance', None)
        if distance is not None and distance < 0.5:
            emit('stop_signal', {'reason': 'ultrasonic', 'distance': distance})
            should_stop = True

        # 자동차로 속도와 조향각 전송
        if not should_stop:
            # 조향각에 따라 속도를 조정
            if abs(smoothed_steering_angle_deg - 90) < 40:  # 조향각이 중심에 가까우면 직진
                speed = 0.3
            else:  # 조향각이 중심에서 벗어나면 회전
                speed = 0.2
            emit('steering_angle', {'angle': smoothed_steering_angle_deg})
            emit('speed', {'speed': speed})
            #print(f"속도 명령 전송: {speed}")
        else:
            # 정지 신호를 보냈으므로 속도 0으로 설정
            emit('speed', {'speed': 0})
            #print(f"정지정지")
        # 이미지 저장
        save_debug_images(frame, roi_frame, src, top_view, cropped_top_view, sliding_window_result)

        # 디버깅용 디스플레이
        cv2.imshow("Top View", top_view)
        cv2.imshow("Cropped Top View", cropped_top_view)
        cv2.imshow("Sliding Window Result", sliding_window_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            socketio.stop()

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

    # src 좌표 설정 (ROI 기준으로 정의)
    src = np.float32([
        [width * 0.35, lower_bound],  # Top-left
        [width * 0.7, lower_bound],   # Top-right
        [width * 1.0, height],        # Bottom-right
        [width * 0.0, height]         # Bottom-left
    ])

    # dst 좌표 설정 (Top View 기준)
    dst = np.float32([
        [width * 0.35, 0],             # Top-left
        [width * 0.7, 0],              # Top-right
        [width * 1.0, height],         # Bottom-right
        [width * 0.0, height]          # Bottom-left
    ])

    # 원근 변환 수행
    M = cv2.getPerspectiveTransform(src, dst)
    top_view = cv2.warpPerspective(frame, M, (width, height), flags=cv2.INTER_LINEAR)
    
    # 하단의 불필요한 부분을 잘라내기 (cutoff 적용)
    cutoff = int(height * 0.85)  # 하단 15%를 제거
    cropped_top_view = top_view[:cutoff, :]  # 상단부터 cutoff까지만 유지

    return roi_frame, src, top_view, cropped_top_view

def apply_sliding_window(img):
    """
    HLS 기반 이진화와 슬라이딩 윈도우를 사용하여 차선 또는 피처를 감지.
    """
    height, width = img.shape[:2]
    window_height = 50
    num_windows = height // window_height
    window_margin = 100  # 탐색 범위 마진

    # 1. HLS 색 공간으로 변환
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 2. HLS에서 흰색 차선을 위한 범위 설정 및 이진화
    lower_white_hls = np.array([0, 210, 0])  # HLS에서 흰색 차선의 범위
    upper_white_hls = np.array([180, 255, 255])
    binary = cv2.inRange(hls, lower_white_hls, upper_white_hls)

    # 3. 히스토그램을 사용하여 관심 영역 찾기
    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    midpoint = len(histogram) // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # 4. 슬라이딩 윈도우 시각화 및 차선 중심 계산
    output_img = np.dstack((binary, binary, binary))
    left_current = left_base
    right_current = right_base

    # 차선 감지 실패 시 None 반환
    if left_current == 0 and right_current == midpoint:
        return output_img, None

    lane_center = (left_current + right_current) / 2  # 차선의 중앙 계산

    for window in range(num_windows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        # 왼쪽 차선 윈도우 설정 및 시각화
        win_xleft_low = max(left_current - window_margin, 0)
        win_xleft_high = min(left_current + window_margin, width)
        cv2.rectangle(output_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)

        # 오른쪽 차선 윈도우 설정 및 시각화
        win_xright_low = max(right_current - window_margin, 0)
        win_xright_high = min(right_current + window_margin, width)
        cv2.rectangle(output_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 왼쪽 윈도우 내의 흰색 픽셀 찾기
        left_nonzero = binary[win_y_low:win_y_high, win_xleft_low:win_xleft_high].nonzero()
        left_nonzero_x = left_nonzero[1] + win_xleft_low

        # 오른쪽 윈도우 내의 흰색 픽셀 찾기
        right_nonzero = binary[win_y_low:win_y_high, win_xright_low:win_xright_high].nonzero()
        right_nonzero_x = right_nonzero[1] + win_xright_low

        # 흰색 픽셀이 충분하다면 중앙 위치 업데이트
        if len(left_nonzero_x) > 50:
            left_current = int(np.mean(left_nonzero_x))
        if len(right_nonzero_x) > 50:
            right_current = int(np.mean(right_nonzero_x))

    return output_img, lane_center

def save_debug_images(frame, roi_frame, src, top_view, cropped_top_view, sliding_window_result):
    """디버깅용 이미지를 저장."""
    # 원본 이미지 저장
    cv2.imwrite(f"{OUTPUT_DIR}/original.jpg", frame)

    # ROI 처리된 이미지 저장
    cv2.imwrite(f"{OUTPUT_DIR}/_roi.jpg", roi_frame)

    # Top View 저장
    cv2.imwrite(f"{OUTPUT_DIR}/_top_view.jpg", top_view)
    cv2.imwrite(f"{OUTPUT_DIR}/cropped_top_view.jpg", cropped_top_view)

    # 슬라이딩 윈도우 결과 저장
    cv2.imwrite(f"{OUTPUT_DIR}/sliding_window_result.jpg", sliding_window_result)

    # src 좌표 시각화 및 저장
    frame_with_src = frame.copy()
    for point in src:
        cv2.circle(frame_with_src, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # src는 빨간 점으로 표시
    cv2.imwrite(f"{OUTPUT_DIR}/src_visualization.jpg", frame_with_src)

    #print(f"이미지가 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
