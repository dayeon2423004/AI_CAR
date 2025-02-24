import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# 🚗 각도 클래스로 변환하는 함수
def angle_to_class(angle):
    if -5 <= angle <= 5:
        return 0  # 직진
    elif 5 < angle <= 15:
        return 1  # 좌회전 약
    elif angle > 15:
        return 2  # 좌회전 강
    elif -15 <= angle < -5:
        return 3  # 우회전 약
    else:
        return 4  # 우회전 강

# 데이터셋 정의 (분류용)
class LineImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform
    
    def _parse_image_name(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        parts = base_name.split('_')
        angle = float(parts[1])  # 파일명에서 각도 추출
        return angle_to_class(angle)  # 각도를 클래스 번호로 변환

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self._parse_image_name(image_name)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_files)

# 데이터 전처리
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터 로드
image_folder = r"c:\\Users\\USER\\Desktop\\최종폴더"
dataset = LineImageDataset(image_folder=image_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # 14000장에 맞게 배치 사이즈 증가

# ViT 모델 정의
class ViTAngleClassifier(nn.Module):
    def __init__(self):
        super(ViTAngleClassifier, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=5
        )
    
    def forward(self, x):
        return self.model(x).logits

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTAngleClassifier().to(device)

if torch.cuda.is_available():
    print("GPU 사용 가능: ", torch.cuda.get_device_name(0))
else:
    print("GPU 사용 불가, CPU 사용 중")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # ViT는 더 작은 학습률이 적합

# 학습 루프
epochs = 50  # 더 많은 데이터에 맞게 에폭 수 증가
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}, Accuracy: {accuracy:.2f}%")

# 모델 저장
model_save_path = "C:\\Users\\USER\\Desktop\\vit_angle_classification_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")
