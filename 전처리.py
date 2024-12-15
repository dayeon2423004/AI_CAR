import cv2
import numpy as np
import os
import glob

def process_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"{image_path} 이미지를 로드할 수 없습니다.")
        return None, None
    
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

# 이미지가 있는 폴더 경로 설정
input_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 여기에 폴더 경로를 입력하세요
output_folder = "C:\\Users\\USER\\Desktop\\copy"  # 결과를 저장할 폴더 경로

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 이미지 파일에 대해 처리
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 형식 확인
        image_path = os.path.join(input_folder, filename)
        
        # 이미지 처리
        mask_cleaned, original_image = process_image(image_path)

        if mask_cleaned is not None:
            # 결과 이미지 저장
            mask_output_path = os.path.join(output_folder, f'mask_{filename}')
            cv2.imwrite(mask_output_path, mask_cleaned)
            print(f'Processed and saved: {mask_output_path}')

            # 원본 이미지 저장 (선택 사항)
            original_output_path = os.path.join(output_folder, f'original_{filename}')
            cv2.imwrite(original_output_path, original_image)
            print(f'Saved original image: {original_output_path}')
