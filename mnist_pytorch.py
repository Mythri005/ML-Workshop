import torch  #venv is essential because versions may change or mismatch
import torch.nn as nn 
import torch.nn.functional as func
from torch import optim
from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

Tform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.5),(0.5))])
#downloaded the dataset and tranform is applied
train_dataset = MNIST(root = './data', train = True, download=True, transform = Tform)
test_dataset = MNIST(root='./data', train = False, download=True, transform = Tform)

#loading the dataset
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()         #instantiate my super class
        self.fc1=nn.Linear(28*28, 256)            #fully connected
        self.fc2=nn.Linear(256, 128)
        self.fc3=nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)           #Flatten can also be used x = x.flatten()
        #x = x.flatten()   flatten or view is to convert the 2D to 1D because MLP can't take 2D
        x = self.fc1(x)
        x = func.relu(x)               #Or can combine the function moving to the first layer func.relu(self.fc1(x)) and the activation function
        x = func.relu(self.fc2(x))     #It is going through the second layer and the activation function
        x = self.fc3(x)                #It applies to it at the last layer
        return x

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)    #lr -- learning rate
#Optimizer should update all the parameters we needed


#MOdel in by defalut train mode
num_epoch = 10
for epoch in range(num_epoch):
    running_loss = 0.0                      #It keep tracks of loss for every epoch and finally added to get the epoch level loss
    for image, label in train_loader:
        output = model(image)
        loss = criterion(output, label)          #For every load one of batch data is loaded and loss is calculated for the batch
        loss.backward()                          #For back propogation
        optimizer.step()                         #Update the base parameters
        optimizer.zero_grad()                    #For every gradient is made to 0
        running_loss+=loss.item()                #Loss is tensor and we want to extract a scaler value hence use the item()
    print(f'Epoch[{(epoch+1)}], Loss: {running_loss/len(train_loader)}')
    #Model is training is seen by the loss values
    #If the loss values are decreasing then it shows that model is learning


model.eval()
with torch.no_grad():
    correct = 0.0
    total = 0.0
    for image, label in test_loader:
        output=model(image)
        _, predict = torch.max(output, 1)           #Column wise sum and _, ignore the value and keep the index
        correct+=(predict==label).sum().item()   #Item is used to get the scaler sum value
        total+=label.size(0)
        accuracy = (correct/total)*100
    print(f"Accuracy - {accuracy}")


