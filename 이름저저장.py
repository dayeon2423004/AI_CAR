import os
import csv
import shutil

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\120"  # 원본 이미지 폴더
output_folder = "C:\\Users\\USER\\Desktop\\processed_images"  # 처리된 이미지를 저장할 폴더
csv_file_path = "C:\\Users\\USER\\Desktop\\output1.csv"  # 결과 CSV 파일 경로

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSV 파일에서 데이터 읽기
def read_csv_data(csv_file_path):
    data = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            image_name, angle, speed = row
            data[image_name] = (angle, speed)
    return data

# 이미지 존재 여부 확인 및 복사
def check_and_copy_images(image_folder, output_folder, csv_data):
    for index, image_name in enumerate(os.listdir(image_folder)):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_name)
            if image_name in csv_data:
                angle, speed = csv_data[image_name]
                new_image_name = f"{index + 1}_{angle}_{speed}.jpg"  # 고유번호, 각도, 속도
                new_image_path = os.path.join(output_folder, new_image_name)
                shutil.copy(image_path, new_image_path)  # 이미지 복사
                print(f"Copied: {new_image_path}")
            else:
                print(f"{image_name}: 찾지 못했습니다.")

# 실행
try:
    csv_data = read_csv_data(csv_file_path)  # CSV 데이터 읽기
    check_and_copy_images(image_folder, output_folder, csv_data)  # 이미지 존재 여부 확인 및 복사
except Exception as e:
    print(f"Error: {e}")
