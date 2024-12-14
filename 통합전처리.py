import cv2
import numpy as np
import glob
import os
import math
import csv
import shutil
import pandas as pd

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지 폴더
output_folder = "C:\\Users\\USER\\Desktop\\전처리 완료 폴더2"  # 결과 저장 폴더
output_csv = "C:\\Users\\USER\\Desktop.csv"  # CSV 파일 경로
save_folder = "C:\\Users\\USER\\Desktop\\전처리 완료 폴더"  # 이름 변경된 이미지 저장 폴더

# 결과 저장 폴더가 없으면 생성
mask_folder = os.path.join(output_folder, 'Masks')
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

# 이미지 파일 경로 리스트
image_paths = glob.glob(f'{image_folder}/*.jpg')

# 이미지 전처리 및 마스크 생성 함수
def preprocess_image(image_path):
    """이미지를 전처리하고 마스크를 생성."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"{image_path} 이미지를 로드할 수 없습니다.")
        return None
    
    # 이미지 크기 조정
    image = cv2.resize(image, (640, 480))
    
    # 색상 마스크 생성
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 모폴로지 연산
    kernel = np.ones((7, 7), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    return mask_cleaned, image

# 이미지 각도 계산 함수
def calculate_angles(image_path):
    """이미지에서 직선을 검출하고 각도를 계산."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 엣지 감지 (Canny Edge Detection)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # 각도를 계산 (라디안을 도로 변환)
            angle = math.degrees(theta)
            angle = angle % 180
            angles.append(angle)

    # 평균 각도 계산 및 90도 추가
    if angles:
        mean_angle = np.mean(angles)
        rounded_angle = round(mean_angle / 20) * 20
        adjusted_angle = (rounded_angle + 90) % 180  # 90도 추가
        return adjusted_angle
    return None

# 이미지 전처리 및 각도 계산 후 CSV 저장
def process_images_and_save_csv():
    """이미지를 전처리하고 각도 및 속도를 계산하여 CSV에 저장."""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Average Angle", "Speed"])

        for image_path in image_paths:
            # 이미지 전처리
            mask_cleaned, image = preprocess_image(image_path)
            if mask_cleaned is not None:
                base_filename = os.path.basename(image_path)
                output_path = os.path.join(mask_folder, base_filename)
                # JPEG 압축을 85% 품질로 적용하여 저장
                cv2.imwrite(output_path, mask_cleaned, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # 각도 계산
                angle = calculate_angles(image_path)
                speed = 40  # 속도는 40으로 고정

                if angle is not None:
                    writer.writerow([base_filename, angle, speed])
                    print(f"Processed {base_filename}: Angle = {angle}, Speed = {speed}")
                else:
                    print(f"{base_filename}에서 각도를 계산할 수 없습니다.")

# 이미지 이름 변경 및 저장 (모폴로지 연산 후 이미지)
def rename_images_with_metadata():
    """CSV 파일을 기반으로 모폴로지 연산 후 이미지 이름을 변경하여 저장."""
    data = pd.read_csv(output_csv)
    os.makedirs(save_folder, exist_ok=True)

    for idx, row in data.iterrows():
        original_filename = row['Image Name']
        angle = row['Average Angle']
        speed = row['Speed']

        new_id = f"{idx+1:05d}"
        new_filename = f"image_{new_id}_angle_{angle}_speed_{speed}.jpg"

        # 모폴로지 연산 후 이미지 경로
        original_path = os.path.join(mask_folder, original_filename)
        new_path = os.path.join(save_folder, new_filename)

        if os.path.exists(original_path):
            shutil.copy2(original_path, new_path)
            print(f"Renamed and saved: {new_filename}")
        else:
            print(f"Warning: {original_filename} not found in {mask_folder}")

# 실행
try:
    process_images_and_save_csv()
    print(f"Results saved to {output_csv}")
    rename_images_with_metadata()
    print(f"Renamed images saved to {save_folder}")
except Exception as e:
    print(f"Error: {e}")
