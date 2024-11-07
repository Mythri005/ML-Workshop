import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as func
from torchvision.datasets import CIFAR10
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

Tform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = CIFAR10(root = './data', train=True, download=True, transform = Tform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform = Tform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class  CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)        #convolution layer 1, 
        #filter size decreases and no. of filter increases going the layers and 32 fliters are used they are the output channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)       #3 is the image size
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 256)       #128 channels and after padding 4*4 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #Expects 2D image so no need to flatten the image
        x = self.conv1(x)
        x = func.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(func.relu(x))
        x = self.pool(func.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4)     #no. of the output is the n
        #pooling will not be there after the convolution layers in fully connected layers
        x = torch.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)        #No activation function at the output layer or last layer
        return x

model = CNNModel()
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

plt.plot(epoch)
plt.show()