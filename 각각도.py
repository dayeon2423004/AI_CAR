import cv2
import numpy as np
import os
import csv
import math

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\cropped_images"  # 원본 이미지 폴더
output_csv = "C:\\Users\\USER\\Desktop\\output1.csv"  # 결과 저장 CSV 파일 경로

# 이미지에서 직선의 각도를 계산하는 함수 (30도 단위로 계산)
def calculate_angle(image_path):
    """이미지에서 직선의 각도를 계산"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None
    
    # 이미지를 회색조로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ROI 설정: 중앙 부분만 사용
    height, width = gray.shape
    roi = gray[int(height * 0.4):int(height * 0.6), :]  # 중앙 20% 영역

    # Gaussian 블러 적용
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # 엣지 검출
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Hough 변환으로 직선 검출 (HoughLinesP 사용)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            # 각도 계산
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = angle % 180  # 0 ~ 180도 범위로 조정
            angles.append(angle)

    # 평균 각도 계산 후 30도 단위로 반올림
    if angles:
        mean_angle = np.mean(angles)
        rounded_angle = round(mean_angle / 30) * 30  # 30도 단위로 반올림
        return rounded_angle
    return None

# 이미지 이름과 각도를 CSV로 저장하는 함수
def process_images_and_save_csv():
    """이미지에서 각도를 계산하고 속도 40과 함께 CSV에 저장"""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Average Angle", "Speed"])  # 헤더

        for image_name in os.listdir(image_folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(image_folder, image_name)
                
                # 각도 계산
                angle = calculate_angle(image_path)
                
                if angle is not None:
                    # 속도는 40으로 고정
                    speed = 40
                    writer.writerow([image_name, angle, speed])
                    print(f"Processed {image_name}: Angle = {angle}, Speed = {speed}")
                else:
                    print(f"No lines detected in {image_name}")

# 실행
try:
    process_images_and_save_csv()
    print(f"Results saved to {output_csv}")
except Exception as e:
    print(f"Error: {e}")
