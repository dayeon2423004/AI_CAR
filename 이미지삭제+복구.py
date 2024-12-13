import cv2
import os
import shutil

# 이미지 탐색 및 조작 기능
image_folder = "/home/kimdayeon/Desktop/copied_images"  # 이미지가 저장된 폴더 경로
deleted_folder = "/home/kimdayeon/Desktop/deleted_images"  # 삭제된 파일을 저장할 폴더 경로
os.makedirs(deleted_folder, exist_ok=True)  # 삭제된 이미지 폴더 생성
deleted_files = []  # 삭제된 파일 경로를 저장하는 리스트

def browse_images(folder_path):
    """폴더 내 이미지를 탐색하며 키 입력에 따라 동작."""
    print("a : 이전으로, d : 다음으로, s : 이미지 삭제, w : 삭제된 파일 복구, q : 종료")
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    images.sort()  # 파일 이름 기준 정렬

    if not images:
        print("폴더에 이미지가 없습니다.")
        return

    index = 0

    while 0 <= index < len(images):
        image_path = os.path.join(folder_path, images[index])
        img = cv2.imread(image_path)

        if img is None:
            print(f"이미지 로드 실패: {images[index]}")
            index += 1
            continue

        # 이미지 표시
        cv2.imshow("Image Viewer", img)
        key = cv2.waitKey(0)

        if key == ord('d'):  # 다음 이미지
            index += 1
        elif key == ord('a'):  # 이전 이미지
            index -= 1
            if index == 0:
                    print("마지막 장입니다.")
        elif key == ord('s'):  # 이미지 삭제
            print(f"삭제할 이미지: {images[index]}")
            deleted_path = os.path.join(deleted_folder, images[index])  # 삭제된 파일의 경로
            shutil.move(image_path, deleted_path)  # 파일을 삭제된 폴더로 이동
            deleted_files.append(deleted_path)  # 삭제된 파일 경로를 리스트에 추가
            images.pop(index)  # 리스트에서 이미지 삭제
            if index >= len(images):
                index -= 1
        elif key == ord('w'):  # 삭제된 파일 복구
            if deleted_files:
                restore_path = deleted_files.pop()  # 복구할 이미지 경로 가져오기
                restored_name = os.path.basename(restore_path)
                restored_path = os.path.join(folder_path, restored_name)

                shutil.copy2(restore_path, restored_path)  # 파일 복사
                print(f"복구된 이미지: {restored_name}")
            else:
                print("복구할 이미지가 없습니다.")
        elif key == ord('q'):  # 실행 중지
            print("프로그램 종료.")
            break
        elif key == 27:  # ESC 키를 누르면 종료
            break

    cv2.destroyAllWindows()

# 실행
browse_images(image_folder)