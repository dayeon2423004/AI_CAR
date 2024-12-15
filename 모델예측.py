import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# 모델 정의 (저장한 모델과 동일한 구조여야 함)
class AnglePredictionModel(nn.Module):
    def __init__(self):
        super(AnglePredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 불러오기
model = AnglePredictionModel()
model.load_state_dict(torch.load("C:\\Users\\USER\\Desktop\\angle_prediction_model.pth"))
model.eval()  # 평가 모드로 전환

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 검증 이미지 경로 설정
validation_image_path = "C:\\Users\\USER\\Desktop\\frame_20241206_124421.jpg"  # 예시 이미지 경로

# 이미지 로드 및 전처리
image = Image.open(validation_image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # 배치 차원 추가

# 예측 수행
with torch.no_grad():
    output = model(image)
    predicted_angle = output.item()  # 예측된 각도를 가져옵니다.

print(f"Predicted Angle: {predicted_angle:.2f} degrees")
