import os
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 폴더 경로 및 저장할 폴더 경로
input_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지가 있는 폴더
output_folder = "C:\\Users\\USER\\Desktop\\cropped_images"  # 잘라낸 이미지를 저장할 폴더

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 처리 함수
def crop_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 형식 확인
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # 이미지 크기 가져오기
            width, height = image.size

            # 잘라낼 비율 설정
            top_crop_percentage = 0.3  # 상단 30%
            side_crop_percentage = 0.002  # 좌우 0.5%

            # 잘라낼 영역 계산
            left = int(width * side_crop_percentage)  # 좌측 0.5%
            top = int(height * top_crop_percentage)  # 상단 30%
            right = width - int(width * side_crop_percentage)  # 우측 0.5%
            bottom = height  # 아래쪽은 원본 높이 유지

            # 이미지 자르기
            cropped_image = image.crop((left, top, right, bottom))

            # 잘라낸 이미지 저장
            cropped_image_path = os.path.join(output_folder, f"cropped_{filename}")
            cropped_image.save(cropped_image_path)

            print(f"Processed and saved: {cropped_image_path}")

# 이미지 자르기 및 저장 실행
crop_images(input_folder, output_folder)
print("모든 이미지가 처리되었습니다.")
