import warnings
warnings.filterwarnings("ignore")

import os
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO

import torch
import torch.nn as nn

yolo_model = YOLO("runs/detect/train/weights/best.pt")
input_path = r"C:\Users\Sina\Desktop\HamrahTel\Handwritten\test.jpg"
results = yolo_model(input_path)

for result in results:
    boxes = result.boxes

    image = cv.cvtColor(cv.imread(input_path), cv.COLOR_BGR2RGB)
    xy_xy = list(map(int, boxes.xyxy[1].tolist()))

    cropped_image = image[xy_xy[1]:xy_xy[3], xy_xy[0]:xy_xy[2], :]

    cropped_image = cv.resize(cropped_image, (256, 64))
    cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(16 * 4 * 16, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = {0: "False", 1: "True"}

model = CNN()

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# print("CNN Model loaded successfully!")

target_image = torch.tensor(cropped_image, dtype=torch.float32)
target_image = target_image.unsqueeze(0).unsqueeze(0)
target_image = target_image.to(device)

model.eval()

with torch.no_grad():
    output = model(target_image).squeeze()

probability = output.item()
confidence = 1 - probability

print("Predicted Label:", labels[output.item()])
print("Confidence (%):", confidence * 100)