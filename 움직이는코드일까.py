import cv2
import RPi.GPIO as GPIO
import time
import torch
from torchvision import transforms
import torch.nn as nn

# 서보 및 모터 GPIO 핀 설정
SERVO_PIN = 18       # 서보 모터 제어 핀
MOTOR_ENA = 23       # L298N ENA 핀 (속도 제어)
MOTOR_IN1 = 24       # L298N IN1 핀 (전진)
MOTOR_IN2 = 25       # L298N IN2 핀 (후진)

# 서보 모터 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM
servo.start(7.5)  # 중립 위치 (7.5%)

# 모터 드라이버 설정
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
motor_pwm = GPIO.PWM(MOTOR_ENA, 100)  # 속도 제어 PWM
motor_pwm.start(0)

# 모델 불러오기
class AnglePredictionModel(nn.Module):
    def __init__(self):
        super(AnglePredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictionModel().to(device)
model.load_state_dict(torch.load('angle_prediction_model.pth'))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 서보 모터 회전 함수
def set_servo_angle(angle):
    duty = 2.5 + (angle / 18.0)  # 각도를 PWM 값으로 변환 (0~180 -> 2.5%~12.5%)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.1)

# DC 모터 제어 함수
def set_motor_speed(speed=40):
    GPIO.output(MOTOR_IN1, GPIO.HIGH)  # 전진 방향
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)  # 속도 설정 (0~100%)

# 실시간 주행 루프
try:
    cap = cv2.VideoCapture(0)  # 카메라 연결 (0번 장치)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다!")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다!")
            break

        # OpenCV에서 읽은 프레임을 PIL 이미지로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # 이미지 전처리
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 모델 예측
        with torch.no_grad():
            predicted_angle = model(image_tensor).item()

        print(f"Predicted angle: {predicted_angle}")

        # RC카 제어
        set_servo_angle(predicted_angle)  # 서보 모터 각도 조정
        set_motor_speed(40)              # 속도 고정

        # 실시간 화면 출력
        cv2.imshow('RC Car Camera', frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("프로그램 종료")
finally:
    cap.release()       # 카메라 정리
    cv2.destroyAllWindows()
    servo.stop()        # 서보 정리
    motor_pwm.stop()    # 모터 정리
    GPIO.cleanup()      # GPIO 정리
