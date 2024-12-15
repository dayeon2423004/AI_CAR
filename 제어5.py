import threading
import time
import subprocess
import keyboard
import cv2
import os
import Jetson.GPIO as GPIO
from flask import Flask, render_template, request, Response

app = Flask(__name__)

# GPIO 경고 비활성화
GPIO.setwarnings(False)

# Set the sudo password as a variable for easy updating
sudo_password = "rowx0097"

# Function to run shell commands with the sudo password
def run_command(command):
    full_command = f"echo {sudo_password} | sudo -S {command}"
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

# Set up GPIO pins for servo and DC motor control
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
servo.start(0)
dc_motor_pwm.start(0)

servo_angle = 0
current_frame = None  # 현재 프레임을 저장할 변수

# Function to stop all motor actions
def stop_all_actions():
    servo.ChangeDutyCycle(0)
    dc_motor_pwm.ChangeDutyCycle(0)
    GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2, GPIO.LOW)

# Function to set DC motor speed and direction
def set_dc_motor_speed(direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        dc_motor_pwm.ChangeDutyCycle(70)
        print("DC Motor moving forward.")
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        dc_motor_pwm.ChangeDutyCycle(70)
        print("DC Motor moving backward.")

# Function to adjust servo position
def adjust_servo(direction):
    global servo_angle
    if direction == "left":
        servo_angle = max(0, servo_angle - 10)
    elif direction == "right":
        servo_angle = min(180, servo_angle + 10)
    servo.ChangeDutyCycle(servo_angle / 18 + 2)
    print(f"Servo angle set to {servo_angle} degrees.")

# 이미지 저장 함수
def save_image():
    global current_frame
    if current_frame is None:
        print("현재 프레임이 없습니다.")
        return

    image_index = 1
    while os.path.exists(f'image_{image_index}.jpg'):
        image_index += 1
    cv2.imwrite(f'image_{image_index}.jpg', current_frame)
    print(f"이미지가 image_{image_index}.jpg로 저장되었습니다.")

# 비디오 캡처 및 저장 스레드
def generate_frames():
    global current_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 중복되지 않는 파일 이름 생성
    video_index = 1
    while os.path.exists(f'output_{video_index}.avi'):
        video_index += 1
    out = cv2.VideoWriter(f'output_{video_index}.avi', fourcc, 20.0, (640, 480))

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_frame = frame  # 현재 프레임 저장
            out.write(frame)

            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 정리
    cap.release()
    out.release()

# RC카 조종 스레드
def control_car():
    try:
        while True:
            if keyboard.is_pressed('w'):  # Forward
                set_dc_motor_speed("forward")
            elif keyboard.is_pressed('s'):  # Backward
                set_dc_motor_speed("backward")
            elif keyboard.is_pressed('a'):  # Left
                adjust_servo("left")
            elif keyboard.is_pressed('d'):  # Right
                adjust_servo("right")
            elif keyboard.is_pressed('q'):  # Forward left
                adjust_servo("left")
                set_dc_motor_speed("forward")  # Move forward while turning left
            elif keyboard.is_pressed('e'):  # Forward right
                adjust_servo("right")
                set_dc_motor_speed("forward")  # Move forward while turning right
            elif keyboard.is_pressed('x'):  # Save image with key 'x'
                save_image()  # 이미지 저장 함수 호출
            else:
                stop_all_actions()  # Stop when no key is pressed
            time.sleep(0.1)  # Delay to prevent excessive CPU usage
    finally:
        stop_all_actions()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

# Flask 서버 실행 함수
def run_flask():
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/control', methods=['POST'])
    def control():
        direction = request.form.get('direction')
        if direction in ["forward", "backward", "stop", "left", "right"]:
            if direction == "forward":
                set_dc_motor_speed("forward")
            elif direction == "backward":
                set_dc_motor_speed("backward")
            elif direction == "stop":
                stop_all_actions()
            elif direction == "left":
                adjust_servo("left")
            elif direction == "right":
                adjust_servo("right")
        return ('', 204)

    app.run(host='0.0.0.0', port=5000)

# 스레드 생성
car_thread = threading.Thread(target=control_car)
flask_thread = threading.Thread(target=run_flask)

# 스레드 시작
car_thread.start()
flask_thread.start()

# 메인 스레드에서 다른 작업을 수행할 수 있음
car_thread.join()  # 자동차 제어 스레드가 종료될 때까지 대기
flask_thread.join()  # Flask 서버 스레드가 종료될 때까지 대기
