import os
import re
import matplotlib.pyplot as plt

# 이미지 폴더 경로
image_folder = "C:\\Users\\USER\\Desktop\\copy"  # 이미지를 저장한 폴더 경로

def extract_angle_from_filename(filename):
    """파일 이름에서 각도 추출 (angle_로 시작하는 형식)"""
    match = re.search(r'angle_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def plot_angle_distribution(folder_path):
    """폴더 내 모든 이미지 파일에서 각도를 추출하고 분포를 그린다."""
    angles = []

    # 폴더 내 모든 파일을 순회하며 각도 추출
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            angle = extract_angle_from_filename(image_name)
            if angle is not None:
                angles.append(angle)

    # 각도 분포도 그리기
    if angles:
        plt.figure(figsize=(8, 6))
        plt.hist(angles, bins=18, range=(0, 180), color='skyblue', edgecolor='black')
        plt.title('Distribution of Angles')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    else:
        print("No angles detected in image filenames.")

# 실행
plot_angle_distribution(image_folder)
