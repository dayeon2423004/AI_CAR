import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# CSV 파일 경로 설정
csv_file = "C:\\Users\\USER\\Desktop\\data.csv"  # CSV 파일 경로
image_dir = "C:\\Users\\USER\\Desktop\\sorted_images"  # 이미지 파일이 저장된 디렉토리 경로

# Custom Dataset 정의
class LineTrackingDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.angles = []
        self.speeds = []
        
        # CSV 파일에서 데이터 읽기
        data = pd.read_csv(csv_file)

        for _, row in data.iterrows():
            filename = row['filename']
            angle = row['angle']
            speed = row['speed']
            
            # 각도와 속도가 비어 있거나 잘못된 경우 건너뜀
            if pd.isna(angle) or pd.isna(speed):
                print(f"Skipping file {filename}: Angle or speed is missing")
                continue
            
            try:
                # 이미지 읽기
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping file {filename}: Image not found")
                    continue
                
                img = cv2.resize(img, (128, 128))  # 이미지 크기 조정 (128x128)
                self.images.append(img)
                self.angles.append(float(angle))
                self.speeds.append(float(speed))
            except ValueError as e:
                print(f"Skipping file {filename}: {e}")
                continue
        
        # 확인용 디버깅 출력
        print(f"Total images loaded: {len(self.images)}")
        print(f"Angles loaded: {self.angles[:5]}")  # 첫 5개 각도 출력
        print(f"Speeds loaded: {self.speeds[:5]}")  # 첫 5개 속도 출력
        
        # numpy array로 변환
        self.images = np.array(self.images)
        self.angles = np.array(self.angles)
        self.speeds = np.array(self.speeds)
        
        # 정규화
        if len(self.angles) > 0:  # 데이터가 있을 때만 정규화
            self.images = self.images / 255.0  # 이미지 정규화
            self.angle_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.angle_scaled = self.angle_scaler.fit_transform(self.angles.reshape(-1, 1))  # 각도 정규화
            self.speed_scaler = MinMaxScaler(feature_range=(-1, 1))
            self.speed_scaled = self.speed_scaler.fit_transform(self.speeds.reshape(-1, 1))  # 속도 정규화
        else:
            print("No valid angle/speed data found, exiting.")
            raise ValueError("No valid data found for angles and speeds.")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angle_scaled[idx]
        speed = self.speed_scaled[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(angle, dtype=torch.float32), torch.tensor(speed, dtype=torch.float32)

# 이미지에 대한 변환 정의 (데이터 증강 등을 추가할 수 있음)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 데이터셋 준비
dataset = LineTrackingDataset(csv_file, image_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN 모델 정의
class LineTrackingCNN(nn.Module):
    def __init__(self):
        super(LineTrackingCNN, self).__init__()
        
        # CNN 계층
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 완전연결층
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # (128, 128) 크기 이미지
        self.fc2 = nn.Linear(128, 1)  # 각도 예측
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # 평탄화
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 각도 예측
        
        return x

# 모델 인스턴스 생성
model = LineTrackingCNN()

# 손실 함수 및 최적화 알고리즘
criterion = nn.MSELoss()  # 평균 제곱 오차 (회귀 문제)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
epochs = 8
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, angles, speeds in train_loader:
        optimizer.zero_grad()
        
        # 모델에 입력
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, angles)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# 예측 후 역정규화
model.eval()
with torch.no_grad():
    predicted_angles = model(torch.tensor(dataset.images).permute(0, 3, 1, 2).float())
    predicted_angles = predicted_angles.numpy()
    predicted_angles = dataset.angle_scaler.inverse_transform(predicted_angles)

# 예시 출력
print(f"Predicted Angles: {predicted_angles[:5]}")
