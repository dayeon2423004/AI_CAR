import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard  # 키보드 이벤트를 감지하기 위한 라이브러리

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
servo_pin = 33  # PWM-capable pin for servo motor
dc_motor_pwm_pin = 32  # PWM-capable pin for DC motor speed
dc_motor_dir_pin1 = 29  # Direction control pin 1
dc_motor_dir_pin2 = 31  # Direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

# Initial servo angle
servo_angle = 0

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
        dc_motor_pwm.ChangeDutyCycle(100)  # Full speed for forward
        print("DC Motor moving forward.")
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        dc_motor_pwm.ChangeDutyCycle(100)  # Full speed for backward
        print("DC Motor moving backward.")

# Function to adjust servo position
def adjust_servo(direction):
    global servo_angle
    if direction == "left":
        servo_angle = max(0, servo_angle - 10)  # Decrease angle, limit to 0
    elif direction == "right":
        servo_angle = min(180, servo_angle + 10)  # Increase angle, limit to 180
    servo.ChangeDutyCycle(servo_angle / 18 + 2)  # Convert angle to duty cycle
    print(f"Servo angle set to {servo_angle} degrees.")

# Main control loop using keyboard events
try:
    print("Press 'w' to move forward, 's' to move backward, 'a' to stop all actions, 'd' to move right, 'a' to move left, 'q' to quit.")
    
    while True:
        if keyboard.is_pressed('w'):
            set_dc_motor_speed("forward")
            time.sleep(0.1)  # Delay to prevent excessive CPU usage
        elif keyboard.is_pressed('s'):
            set_dc_motor_speed("backward")
            time.sleep(0.1)  # Delay to prevent excessive CPU usage
#        elif keyboard.is_pressed('a'):
#            stop_all_actions()
#            time.sleep(0.1)  # Delay to prevent excessive CPU usage
        elif keyboard.is_pressed('d'):
            adjust_servo("right")
            time.sleep(0.1)  # Delay to prevent excessive CPU usage
        elif keyboard.is_pressed('a'):
            adjust_servo("left")
            time.sleep(0.1)  # Delay to prevent excessive CPU usage
        elif keyboard.is_pressed('q'):
            print("Exiting control.")
            break
        else:
            stop_all_actions()  # Stop if no key is pressed

finally:
    stop_all_actions()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()

