import torch
import torch.nn as nn
from torchvision import transforms

import pandas as pd
from PIL import Image
import time
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

data = pd.read_csv("dataset/trainLabels.csv")
classes = sorted(data['label'].unique())


class CNNCIFAR(nn.Module):
    def __init__(self):
        super().__init__()

        ##input (3,32,32)
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,10)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x,1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x
    

model = CNNCIFAR()
model.load_state_dict(torch.load("model/cifar10Model25.pth", weights_only=True))
model.eval()


image_path = "inputs/bird.png"

image = Image.open(image_path).convert('RGB')
image_arr = np.array(image)
image = transform(image)
image = image.unsqueeze(0)


with torch.no_grad():
    start = time.time()
    outputs = model(image)
    end = time.time()
    #print(end-start)
    _,predicted = torch.max(outputs,1)
    predicted_class_index = predicted.item()

predicted_class_name = classes[predicted_class_index]

plt.figure(figsize=((5,5)))
plt.title(f"class: {predicted_class_name}")
plt.imshow(image_arr)
plt.savefig("resources/inference_image.png")
plt.close

