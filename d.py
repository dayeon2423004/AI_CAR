import cv2
import os

# 경로 설정
image_folder = "C:\\Users\\USER\\Desktop\\sorted_images"  # 원본 이미지 폴더
output_folder = "C:\\Users\\USER\\Desktop\\output_images"  # 결과 저장 폴더

# 결과 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 화살표를 그릴 함수
def draw_arrow_on_image(image, angle):
    """이미지 중앙에 지정된 각도에 맞는 화살표를 그린다."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)  # 이미지 중앙
    
    # 화살표의 길이 및 색상 설정
    arrow_length = 100
    arrow_color = (0, 0, 255)  # 빨간색 화살표
    
    # 각도를 라디안으로 변환
    angle_rad = np.deg2rad(angle)
    
    # 화살표 끝 점 계산 (기준점에서 angle만큼 회전)
    end_point = (int(center[0] + arrow_length * np.cos(angle_rad)),
                int(center[1] - arrow_length * np.sin(angle_rad)))  # 반전된 y축 (opencv에서 y는 아래로 증가)
    
    # 화살표 그리기
    image_with_arrow = cv2.arrowedLine(image.copy(), center, end_point, arrow_color, 5, tipLength=0.05)
    
    return image_with_arrow

# 이미지에 각도 표시하고 카피하는 함수
def process_images_and_copy():
    """이미지에서 각도를 계산하고, 화살표를 그려서 새로운 폴더에 카피하여 저장"""
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_name)
            
            # 이미지를 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to load image: {image_path}")
                continue
            
            # 이미지 이름에서 각도 추출 (예: 'image_00001_angle_30_speed_40.jpg'에서 30 추출)
            parts = image_name.split('_')
            
            # 'angle_30' 형식으로 되어 있다고 가정하고 angle 부분만 추출
            try:
                # 각도 추출
                angle_str = next(part for part in parts if 'angle' in part)
                angle = int(angle_str.split('_')[1])  # 'angle_30'에서 30을 추출
            except (StopIteration, IndexError) as e:
                print(f"Error: Couldn't find 'angle' in image name {image_name}")
                continue
            
            # 중앙에 화살표 그리기
            image_with_arrow = draw_arrow_on_image(image, angle)
            
            # 새 파일 이름 생성
            new_filename = f"arrow_{image_name}"
            new_path = os.path.join(output_folder, new_filename)
            
            # 이미지를 새로운 폴더에 저장
            cv2.imwrite(new_path, image_with_arrow)
            print(f"Saved image with arrow: {new_filename}")

# 실행
try:
    process_images_and_copy()
    print(f"Images with arrows saved to {output_folder}")
except Exception as e:
    print(f"Error: {e}")
