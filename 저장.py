import os
import csv

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지 폴더
output_csv = "C:\\Users\\USER\\Desktop\\output.csv"  # CSV 파일 경로

# CSV 파일 읽기
def rename_images_based_on_csv():
    """CSV 파일을 기반으로 이미지 이름을 고유번호, 각도, 속도로 변경"""
    # CSV 파일 읽기
    with open(output_csv, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # 헤더 읽기
        print(f"CSV Header: {header}")

        # 고유번호 시작
        unique_id = 1

        for row in reader:
            image_name = row[0]  # 원본 이미지 이름
            angle = row[1]  # 각도
            speed = row[2]  # 속도

            # 새 파일 이름 생성 (고유번호 포함)
            new_filename = f"{unique_id:05d}_angle_{angle}_speed_{speed}.jpg"

            # 원본 이미지 경로 및 새 이미지 경로
            original_path = os.path.join(image_folder, image_name)
            new_path = os.path.join(image_folder, new_filename)

            # 파일이 존재하면 이름 변경
            if os.path.exists(original_path):
                # 이름 충돌 방지: 새로운 이름 생성
                counter = 1
                while os.path.exists(new_path):
                    new_filename = f"{unique_id:05d}_angle_{angle}_speed_{speed}_{counter}.jpg"
                    new_path = os.path.join(image_folder, new_filename)
                    counter += 1

                # 이름 변경
                os.rename(original_path, new_path)
                print(f"Renamed: {image_name} -> {new_filename}")
            else:
                print(f"File not found: {image_name}")

            # 고유번호 증가
            unique_id += 1

# 실행
try:
    rename_images_based_on_csv()
    print("All images renamed successfully.")
except Exception as e:
    print(f"Error: {e}")
