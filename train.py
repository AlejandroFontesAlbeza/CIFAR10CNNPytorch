
import pandas as pd
import os
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, random_split
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv("dataset/trainLabels.csv")
classes = sorted(data['label'].unique())

label_to_index = {label : index for index, label in enumerate(classes)}

class CustomCIFARDataset(Dataset):
    def __init__(self, csv_file, label_to_index, img_dir, transform = None):
        
        self.dataFrame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, str(self.dataFrame.iloc[index,0]) + ".png")
        label_str = self.dataFrame.iloc[index,1]
        label = self.label_to_index[label_str]

        image = Image.open(img_name).convert('RGB')


        if self.transform:
            image = self.transform(image)

        return image, label
    

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
    


dataset = CustomCIFARDataset(csv_file="dataset/trainLabels.csv",
                                   label_to_index = label_to_index,
                                   img_dir="dataset/train",
                                   transform=transform)



train_dataset,val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset,batch_size=200, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=200, shuffle=False, num_workers=4)


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

        self.dropout = nn.Dropout(0.25)
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
    

model = CNNCIFAR().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)


epochs = 25
epochs_list = []
train_loss_list = []
val_loss_list = []
accuracy_list = []


for epoch in range(epochs):
    print(f'Epoch : {epoch}')
    epochs_list.append(epoch)
    model.train()
    running_loss = 0

    for inputs, labels in tqdm(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    model.eval()
    running_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs,labels)
            running_val_loss += loss.item()


            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    train_loss = (running_loss / len(train_loader))
    train_loss_list.append(train_loss)
    val_loss = (running_val_loss/len(val_loader))
    val_loss_list.append(val_loss)
    accuracy = ((100 * correct)/total)
    accuracy_list.append(accuracy)

    print(f"training loss: {train_loss}, val loss : {val_loss}, accuracy : {accuracy}")



torch.save(model.state_dict(),"model/cifar10Model.pth")
print("modelo guardado")

plt.figure(figsize=(20,5))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(epochs_list, train_loss_list, label = 'training loss')
ax1.plot(epochs_list, val_loss_list, label = 'validation loss')
ax1.legend()

ax2.plot(epochs_list, accuracy_list, label = 'accuracy %')
ax2.legend()
plt.savefig("resources/grafica.png")
plt.close()