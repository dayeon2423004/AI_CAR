import os
import csv
import re

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지 폴더
output_csv = "C:\\Users\\USER\\Desktop\\output.csv"  # CSV 파일 경로

# 이미지 파일 경로 리스트
image_paths = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 이미지 이름에서 각도 추출 함수
def extract_angle_from_filename(filename):
    """파일 이름에서 각도를 추출 (예: 196273_angle_0_speed_40)"""
    match = re.search(r'_angle_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# 이미지 이름에서 각도를 추출하고 CSV에 저장하는 함수
def process_images_and_save_csv():
    """이미지 이름에서 각도를 추출하고 CSV 파일에 저장"""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Angle", "Speed"])

        for image_name in image_paths:
            # 이미지 이름에서 각도 추출
            angle = extract_angle_from_filename(image_name)
            if angle is not None:
                speed = 40  # 속도는 40으로 고정
                writer.writerow([image_name, angle, speed])
                print(f"Processed {image_name}: Angle = {angle}, Speed = {speed}")

# 실행
try:
    process_images_and_save_csv()
    print(f"Results saved to {output_csv}")
except Exception as e:
    print(f"Error: {e}")
