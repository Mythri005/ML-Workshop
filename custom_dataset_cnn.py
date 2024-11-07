import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.transforms import transforms
from torch import optim
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

Tform = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
dataset = ImageFolder(root=r"C:\Users\mythr\Downloads\archive\PlantVillage", transform = Tform)

#Split the dataset into train and validation or test set
train_ratio = 0.8
train_size = int(train_ratio*len(dataset))
test_size = len(dataset)-train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#Also can write 0.8 and 0.2 in place of train and test sizes

#print(train_dataset[0][0][0].shape)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32,3, stride=2)
        self.conv2 = nn.Conv2d(32, 64,3)
        self.conv3 = nn.Conv2d(64, 128,3, stride=2)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 15)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.pool(x)
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1, 128*7*7)
        x = torch.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = CNNModel()
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

#Train the model
num_epoch = 10
for epoch in range(num_epoch):
    running_loss = 0.0
    for image, label in train_loader:
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print(f'Epoch-{epoch+1/num_epoch}, Loss: {running_loss/len(train_loader)}')

model.eval()
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for image, label in test_loader:
        output = model(image)
        _, predict = torch.max(output, 1)
        correct += (predict==label).sum().item()
        total +=label.size(0)
        accuracy = (correct/total)*100
    print(f'Accuracy: {accuracy}')