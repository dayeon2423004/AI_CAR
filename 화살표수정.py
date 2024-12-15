import cv2
import numpy as np
import math
import os
import csv
import shutil

# CSV 파일과 이미지 폴더 경로 설정
output_csv = "C:\\Users\\USER\\Desktop\\output.csv"  # 결과를 저장할 CSV 파일 경로
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 이미지가 있는 폴더 경로
output_copy_folder = "C:\\Users\\USER\\Desktop\\copy"  # 이미지 복사본 폴더 경로

def visualize_directions(csv_file, image_folder, copy_folder):
    """CSV에 저장된 각도를 기준으로 복사된 이미지에 화살표를 그려 저장."""
    os.makedirs(copy_folder, exist_ok=True)

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row["Image Name"]
            angle = row["Average Angle"]

            if angle == "" or angle is None:
                continue

            angle = float(angle)
            original_image_path = os.path.join(image_folder, image_name)
            copied_image_path = os.path.join(copy_folder, image_name)

            # 이미지 복사
            shutil.copy2(original_image_path, copied_image_path)

            img = cv2.imread(copied_image_path)

            if img is None:
                continue

            height, width, _ = img.shape
            center = (width // 2, height // 2)

            # 화살표의 끝 좌표 계산
            length = 100  # 화살표 길이
            end_x = int(center[0] + length * math.cos(math.radians(angle)))
            end_y = int(center[1] - length * math.sin(math.radians(angle)))

            # 화살표 그리기
            cv2.arrowedLine(img, center, (end_x, end_y), (0, 0, 255), 3, tipLength=0.2)

            # 덮어쓰지 않고 복사된 이미지에 저장
            cv2.imwrite(copied_image_path, img)
            print(f"Saved annotated image to {copied_image_path}")

# 실행
try:
    visualize_directions(output_csv, image_folder, output_copy_folder)
    print("Visualization complete.")
except Exception as e:
    print(f"Error: {e}")
