import cv2
import numpy as np
import glob
import os

# 이미지 파일 경로 리스트
image_paths = glob.glob("C:\\Users\\USER\\Desktop\\sorted_images\\*.jpg")

output_folder = 'output_images'  # 결과를 저장할 기본 폴더

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 마스크 저장 폴더 생성
mask_folder = os.path.join(output_folder, 'Masks')
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

for image_path in image_paths:
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"{image_path} 이미지를 로드할 수 없습니다.")
        continue

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

    # 윤곽선 찾기
    contours_info = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    # 최종 마스크 생성
    mask_final = np.zeros_like(mask_cleaned)
    height, width = mask_cleaned.shape
    y_threshold = height * 2 / 3

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if y + h >= y_threshold and (aspect_ratio < 0.5 or aspect_ratio > 2.0):
                cv2.drawContours(mask_final, [contour], -1, (255), thickness=cv2.FILLED)

    # 최종 마스크 저장
    base_filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(mask_folder, base_filename), mask_final)

print("모든 마스크가 저장되었습니다.")
