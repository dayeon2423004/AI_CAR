import cv2
import numpy as np
import math
import os
import csv

# 폴더 내 모든 이미지의 각도를 계산하고 CSV에 저장
image_folder = "C:\\Users\\USER\\Desktop\\processed_images"  # 이미지가 있는 폴더 경로
output_csv = "C:\\Users\\USER\\Desktop\\output1.csv"  # 결과를 저장할 CSV 파일 경로

def calculate_angles(image_path):
    """이미지에서 직선을 검출하고 각도를 계산."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # 이미지 로드 실패 시 None 반환

    # 엣지 감지 (Canny Edge Detection)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # 각도를 계산 (라디안을 도로 변환)
            angle = math.degrees(theta)
            # 각도를 0 ~ 180도 범위로 조정
            angle = angle % 180
            angles.append(angle)

    # 평균 각도 계산
    if angles:
        mean_angle = np.mean(angles)
        # 20도 단위로 반올림
        rounded_angle = round(mean_angle / 20) * 20
        # 0 ~ 180도 범위로 조정
        return rounded_angle % 180
    return None

def process_images(folder_path, output_file):
    """폴더 내 모든 이미지를 처리하고 각도를 CSV에 저장."""
    # 출력 경로의 디렉터리가 없으면 생성
    if output_file.strip() == "" or not os.path.basename(output_file):
        raise ValueError("Invalid output file path. Ensure it includes a valid filename.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Average Angle", "Speed"])  # 헤더에 속도 추가

        processed_files = set()

        for image_name in os.listdir(folder_path):
            if image_name in processed_files:
                continue

            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                angle = calculate_angles(image_path)
                speed = 40  # 속도는 40으로 고정
                writer.writerow([image_name, angle, speed])  # 이미지 이름, 각도, 속도 저장
                processed_files.add(image_name)
                print(f"Processed {image_name}: Angle = {angle}, Speed = {speed}")

# 실행
try:
    process_images(image_folder, output_csv)
    print(f"Results saved to {output_csv}")
except ValueError as e:
    print(f"Error: {e}")
