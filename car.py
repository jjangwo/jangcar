import socketio
import cv2
import time
import numpy as np
import atexit
from picamera2 import Picamera2
from board import SCL, SDA
import busio
from gpiozero import DistanceSensor
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo

# 모터 및 서보모터 초기화
MOTOR_M1_IN1 = 15
MOTOR_M1_IN2 = 14
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x5f)  # 기본 주소는 0x5f
pca.frequency = 50

# DC 모터
motor1 = motor.DCMotor(pca.channels[MOTOR_M1_IN1], pca.channels[MOTOR_M1_IN2])
motor1.decay_mode = motor.SLOW_DECAY

# 초음파 센서 설정
Tr = 23
Ec = 24
sensor = DistanceSensor(echo=Ec, trigger=Tr, max_distance=2)  # 최대 감지 거리 2m

# # 서보 모터 설정
# MIN_STEERING_ANGLE = 40
# MAX_STEERING_ANGLE = 140
# servo_steering = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2400, actuation_range=160)

# # 카메라 서보 초기화
# camera_left_right = servo.Servo(pca.channels[2], min_pulse=500, max_pulse=2400, actuation_range=160)
# camera_left_right.angle = 90
camera_up_down = servo.Servo(pca.channels[4], min_pulse=500, max_pulse=2400, actuation_range=160)
camera_up_down.angle = 145

# 앞바퀴 조향
servo_steering = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2400, actuation_range=160)

# 프로그램 종료 시 실행되는 함수
def cleanup():
    print("프로그램 종료 중, 모터 초기화 중...")
    motor1.throttle = 0
    servo_steering.angle = 90
    pca.deinit()
    print("모터 초기화 완료.")

atexit.register(cleanup)

# 소켓 클라이언트 초기화
sio = socketio.Client()

@sio.event
def connect():
    print("서버에 연결되었습니다.")

@sio.event
def disconnect():
    print("서버와의 연결이 종료되었습니다.")

@sio.on('registration_success')
def on_registration_success(data):
    car_number = data['car_number']
    print(f"자동차 등록 성공: 자동차 번호 {car_number}")

@sio.on('registration_failed')
def on_registration_failed(data):
    print("자동차 등록 실패:", data['message'])

@sio.on('error')
def on_error(data):
    print("에러 발생:", data['message'])

@sio.on('steering_angle')
def on_steering_angle(data):
    angle = data['angle']
    if 0 <= angle <= 180:
        servo_steering.angle = angle
        print(f"서버로부터 받은 조향각: {angle}도")
        time.sleep(0.25)

@sio.on('speed')
def on_speed(data):
    speed = data['speed']
    motor1.throttle = speed
    print(f"서버로부터 받은 속도: {speed}")

@sio.on('stop_signal')
def on_stop_signal(data):
    motor1.throttle = 0
    reason = data.get('reason', 'unknown')
    if reason == 'person_detected':
        print("사람이 감지되어 자동차를 정지합니다.")
    else:
        print("알 수 없는 이유로 자동차를 정지합니다.")

def main():
    try:
        sio.connect('http://192.168.10.25:5000')
    except Exception as e:
        print("서버에 연결할 수 없습니다:", e)
        return

    # Picamera2 설정
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        distance = sensor.distance
        sio.emit('frame', {'frame': buffer.tobytes(), 'distance': distance})
        time.sleep(0.1)

if __name__ == '__main__':
    main()