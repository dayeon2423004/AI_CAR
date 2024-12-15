import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# 데이터셋 정의
class LineImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform

    def _parse_image_name(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        parts = base_name.split('_')
        angle = float(parts[2])  # 'angle_150'에서 150 추출
        speed = float(parts[4])  # 'speed_40'에서 40 추출
        return angle, speed

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        angle, speed = self._parse_image_name(image_name)
        return image, torch.tensor(angle, dtype=torch.float32)  # 라벨을 float32로 반환

    def __len__(self):
        return len(self.image_files)

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터 로드
image_folder = "C:\\Users\\USER\\Desktop\\copy"
dataset = LineImageDataset(image_folder=image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 정의
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

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, angles in data_loader:
        images, angles = images.to(device), angles.to(device).view(-1, 1).float()  # 데이터 유형 변환 추가

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, angles)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}")

# 모델 저장
model_save_path = "C:\\Users\\USER\\Desktop\\angle_prediction_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")
