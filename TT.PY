import cv2
import numpy as np
import os

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지 폴더
output_folder = "C:\\Users\\USER\\Desktop\\processed_images"  # 결과 저장 폴더
mask_folder = os.path.join(output_folder, 'Masks')  # 마스크 이미지 저장 폴더

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

# 이미지 전처리 및 마스크 생성 함수
def preprocess_image(image_path):
    """이미지를 전처리하고 마스크를 생성."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: {image_path} 이미지를 로드할 수 없습니다.")
        return None
    
    print(f"Processing image: {image_path}")
    
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

    print(f"Mask created for: {image_path}")
    return mask_cleaned

# 이미지 전처리 및 마스크 저장
def process_and_save_masks():
    """이미지를 전처리하고 마스크를 저장."""
    image_files = os.listdir(image_folder)
    for filename in image_files:
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            mask_cleaned = preprocess_image(image_path)
            if mask_cleaned is not None:
                mask_filename = f"mask_{filename}"
                mask_path = os.path.join(mask_folder, mask_filename)
                
                # 저장 경로 확인
                print(f"Saving to path: {mask_path}")
                print(f"Mask shape: {mask_cleaned.shape}, dtype: {mask_cleaned.dtype}")
                
                # 이미지 저장
                success = cv2.imwrite(mask_path, mask_cleaned, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    print(f"Saved mask image: {mask_filename}")
                else:
                    print(f"Failed to save mask image: {mask_filename} at {mask_path}")

# 실행
process_and_save_masks()
print("All masks processed.")
