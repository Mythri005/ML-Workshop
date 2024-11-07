import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image 
import gradio as gr

#Load pre-trained model and fine tune it on the target dataset
model = models.resnet18(pretrained=True)
# model.fc=nn.Linear(model.fc.in_features, 10)
model.fc = nn.Linear(512, 15)

Tform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomRotation(10)
                            ])

class_names = ['class1','class2','class3','class4','class5','class6','class7','class8','class9','class10']
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

print(model)

def predict(img):
    img = Tform(img).unsqueeze(0)      #unsqueeze increases 1 dimension 
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        return predicted_class
    
#Setup gradio interface Django can also used Gradio is the newer version
interface = gr.Interface(
    fn = predict, 
    inputs=gr.Image(type='pil'),
    outputs = 'text',
    title = 'CIFAR dataset prediction',
    description = 'Upload an image to get its class predicted.'
)

interface.launch(share=True)


