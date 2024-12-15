import cv2
import numpy as np
import os
from pathlib import Path

# 입력 및 출력 경로 설정
input_folder = "C:\\Users\\USER\\Desktop\\120"  # 증강할 특정 각도 이미지 폴더
output_folder = "C:\\Users\\USER\\Desktop\\120"    # 증강된 이미지 저장 폴더

# 출력 폴더 생성
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 랜덤 회전 함수
def random_rotation(image, angle_range=(-30, 30)):
    """이미지를 랜덤으로 회전."""
    height, width = image.shape[:2]
    angle = np.random.uniform(*angle_range)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

# 밝기 조절 함수
def adjust_brightness(image, brightness_range=(0.7, 1.3)):
    """이미지 밝기를 조절."""
    brightness_factor = np.random.uniform(*brightness_range)
    return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

# 크기 조정 및 패딩 함수
def resize_with_padding(image, scale_range=(0.8, 1.2), target_size=None):
    """이미지를 랜덤 크기로 조정한 후 패딩."""
    height, width = image.shape[:2]
    scale_factor = np.random.uniform(*scale_range)
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    resized = cv2.resize(image, (new_width, new_height))

    # 패딩 계산 (음수 방지)
    pad_top = max(0, (height - new_height) // 2)
    pad_left = max(0, (width - new_width) // 2)
    pad_bottom = max(0, height - new_height - pad_top)
    pad_right = max(0, width - new_width - pad_left)

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded[:height, :width]

# 증강 처리 함수
def augment_image(image, output_path, start_number, augment_count=5):
    """이미지를 다양한 방식으로 증강하여 저장."""
    for i in range(augment_count):
        augmented = image.copy()

        # 랜덤 회전
        augmented = random_rotation(augmented)

        # 밝기 조절
        augmented = adjust_brightness(augmented)

        # 크기 조정 및 패딩
        augmented = resize_with_padding(augmented)

        # 랜덤 플립
        if np.random.rand() > 0.5:
            augmented = cv2.flip(augmented, 1)  # 좌우 플립

        # 저장
        file_number = start_number + i  # 파일 이름 숫자 시작
        output_filename = os.path.join(output_path, f"{file_number:05d}.jpg")
        cv2.imwrite(output_filename, augmented)
        print(f"증강 이미지 저장 완료: {output_filename}")

# 메인 실행
if __name__ == "__main__":
    start_number = 27014  # 시작 번호 설정
    augment_count = 5  # 각 이미지당 생성할 증강 데이터 수

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path)

            if image is None:
                print(f"{filename}를 열 수 없습니다.")
                continue

            output_path = os.path.join(output_folder, os.path.splitext(filename)[0])
            Path(output_path).mkdir(parents=True, exist_ok=True)  # 출력 서브폴더 생성

            augment_image(image, output_path, start_number, augment_count=augment_count)
            start_number += augment_count  # 다음 이미지의 시작 번호를 업데이트
