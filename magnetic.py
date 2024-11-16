from flask import Flask, request
from flask_socketio import SocketIO, emit
import threading
import cv2
import numpy as np
import torch
import time

app = Flask(__name__)
socketio = SocketIO(app)

# 자동차 정보를 저장할 딕셔너리: {sid: 자동차 번호}
cars = {}
car_number = 1
lock = threading.Lock()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

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
    sid = request.sid
    if sid not in cars:
        emit('error', {'message': '자동차가 등록되지 않았습니다.'})
        return
    car_num = cars[sid]
    # 프레임 디코딩
    img_data = data['frame']
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 라인 검출 및 조향각 계산
    processed_frame, steering_angle = process_line(frame.copy(), f'frames/{car_num}')
    cv2.imwrite(f'frames/{car_num}_processed_frame.jpg', processed_frame)

    # 하단 절반을 남기고 상단 절반 제거 (YOLO 검출용)
    height = frame.shape[0]
    cropped_frame = frame[height//2:, :]  # 이미지의 하단 절반을 남김
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

    # 초음파 센서 거리 확인
    distance = data.get('distance', None)
    if distance is not None and distance < 0.5:
        emit('stop_signal', {'reason': 'ultrasonic', 'distance': distance})
        should_stop = True

    # 자동차로 속도와 조향각 전송
    if not should_stop:
        # 조향각에 따라 속도를 조정
        if abs(steering_angle - 60) < 10:  # 조향각이 중심에 가까우면 직진
            speed = 0.3
        else:  # 조향각이 중심에서 벗어나면 회전
            speed = 0.4
        emit('steering_angle', {'angle': steering_angle})
        emit('speed', {'speed': speed})
    else:
        # 정지 신호를 보냈으므로 속도 0으로 설정
        emit('speed', {'speed': 0})

    # 프레임을 화면에 보여줍니다.
    cv2.imshow(f'자동차 {car_num} 원본 프레임', frame)
    cv2.imshow(f'자동차 {car_num} 라인 검출 프레임', processed_frame)
    cv2.imshow(f'자동차 {car_num} YOLO 프레임', yolo_frame)
    cv2.waitKey(1)

# 조향각 계산 함수 (업데이트된 코드)
def process_line(frame, frames_dir):
    # Step index for naming frames
    step = 0

    # 0번 프레임 저장: 원본 이미지
    cv2.imwrite(f'{frames_dir}/{step}_original_frame.jpg', frame)
    # 주석: 0번 프레임은 원본 이미지를 나타냅니다.
    step += 1

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1번 프레임 저장: 그레이스케일 이미지
    cv2.imwrite(f'{frames_dir}/{step}_grayscale_frame.jpg', gray)
    # 주석: 1번 프레임은 그레이스케일 이미지를 나타냅니다.
    step += 1

    # 가우시안 블러
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 2번 프레임 저장: 블러 처리된 이미지
    cv2.imwrite(f'{frames_dir}/{step}_blurred_frame.jpg', blur)
    # 주석: 2번 프레임은 가우시안 블러가 적용된 이미지를 나타냅니다.
    step += 1

    # 이진화 (흰색 라인 검출)
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    # 3번 프레임 저장: 이진화된 이미지
    cv2.imwrite(f'{frames_dir}/{step}_thresholded_frame.jpg', thresh)
    # 주석: 3번 프레임은 이진화된 이미지를 나타냅니다.
    step += 1

    # 관심 영역 설정 (하단 절반만 사용)
    height, width = thresh.shape
    mask = np.zeros_like(thresh)
    polygon = np.array([[
        (0, height),
        (0, height//2),
        (width, height//2),
        (width, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(thresh, mask)
    # 4번 프레임 저장: 관심 영역이 적용된 이미지
    cv2.imwrite(f'{frames_dir}/{step}_roi_frame.jpg', roi)
    # 주석: 4번 프레임은 관심 영역이 적용된 이미지를 나타냅니다.
    step += 1

    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    steering_angle = 60  # 기본 조향각 (직진)

    left_lines = []
    right_lines = []
    if contours:
        for contour in contours:
            # 윤곽선의 경계 사각형을 구합니다.
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            if center_x < width // 2:
                left_lines.append(contour)
            else:
                right_lines.append(contour)

        # 왼쪽 및 오른쪽 차선의 가장 큰 윤곽선을 찾습니다.
        left_contour = max(left_lines, key=cv2.contourArea) if left_lines else None
        right_contour = max(right_lines, key=cv2.contourArea) if right_lines else None

        if left_contour is not None:
            cv2.drawContours(frame, [left_contour], -1, (255, 0, 0), 3)  # 왼쪽 차선은 파란색으로 그립니다.
        if right_contour is not None:
            cv2.drawContours(frame, [right_contour], -1, (0, 0, 255), 3)  # 오른쪽 차선은 빨간색으로 그립니다.

        # 왼쪽 및 오른쪽 차선의 중심을 계산하여 조향각 설정
        if left_contour is not None and right_contour is not None:
            left_M = cv2.moments(left_contour)
            right_M = cv2.moments(right_contour)
            if left_M['m00'] != 0 and right_M['m00'] != 0:
                left_cx = int(left_M['m10'] / left_M['m00'])
                right_cx = int(right_M['m10'] / right_M['m00'])
                mid_x = (left_cx + right_cx) // 2
                center_offset = mid_x - width // 2
                steering_angle = int(60 - center_offset / (width // 2) * 30)  # 최대 조향각 변화량을 ±50도로 설정
                # 조향각 범위 제한
                steering_angle = max(0, min(120, steering_angle))
                print(f"Calculated steering angle: {steering_angle}")
        elif left_contour is not None:
            # 오른쪽 차선이 안 보일 경우 왼쪽 차선을 따라 이동 (왼쪽으로 크게 회전)
            steering_angle = 35
            print("Right lane missing, turning sharply left.")
        elif right_contour is not None:
            # 왼쪽 차선이 안 보일 경우 오른쪽 차선을 따라 이동 (오른쪽으로 크게 회전)
            steering_angle = 75
            print("Left lane missing, turning sharply right.")

    return frame, steering_angle

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)