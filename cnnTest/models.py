import os.path
from Pytorch_Django_Practice import settings


# Create your models here.
import torch
import torch.nn as nn
import torch.optim as optim
from django.db import models

# CNN 모델 정의 (Hammer, Nipper 분류)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 로드 함수
def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'custom_cnn.pth')
    model = CustomCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
