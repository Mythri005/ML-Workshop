#Fine tuning
import torch 
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
from torchvision.transforms import transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, ImageFolder
import matplotlib.pyplot as plt

Tform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomRotation(10)
                            ])
#For every epoch these transformations are applied
train_dataset = CIFAR10(root = './data', train=True, download=True, transform = Tform)
val_dataset = CIFAR10(root='./data', train=False, download=True, transform = Tform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad=False    #Not training i.e., gradients are not updated
#print(model)

#in_features = model.fc.in_features
model.fc = nn.Linear(512, 15)           #Input 512 is same beacause previos layer may return 512 output to the fc layer
#Not required as optimizer will do
for param in model.fc.parameters():   #Only train the parameters of fc layer
    param.requires_grad=True

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)         #Only update the parameters of fc layer

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

#Validating the model
model.eval()
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for image, label in val_loader:
        output = model(image)
        _, predict = torch.max(output, 1)
        correct += (predict==label).sum().item()
        total +=label.size(0)
        accuracy = (correct/total)*100
    print(f'Accuracy: {accuracy}')

    # if accuracy>best_val_accuracy:
    #     best_val_accuracy = accuracy
torch.save(model.state_dict(), 'best_model.pth')
