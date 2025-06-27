import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os

# project; use resnet50 for face recognition tailored to me specifically
# I want to be able to save my trained model instead of having to retrain it every time
# I only want it to say whether or not it is me. nothing else
# no gui required


# Class inhertiting the Dataset class from torch.utils.data. It assigns the data as an instance of the ImageFolder class with its transformed images
# Also assigns dunder methods to return length and getitem aswell as uses the @property for som shi
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    
# This is the neural netowrk. It takes the different classes, and then initiallizes the resnet50 nerual network to work with my classes
# pretrained is set to true to cut down the training time
# IDK what sequential is; Edit: I DEFINITLY DONT KNOW WHAT SEQUENTIAL IS
class FaceClassifier(nn.Module):
    # 2 Classes because There is only going to be a folder containing me
    def __init__(self, num_classes=1):
        super(FaceClassifier, self).__init__()
        self.base_model = timm.create_model('resnet50', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier=nn.Linear(enet_out_size, num_classes)
# Forward pass function
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)

        return output

# Applies a transform to the given image to resize it
# Iphone Dimentions: Portrait (2316, 3088), Landscape (4023, 3024)
# ykw, screw my gpu we doing full res portrait
transform = transforms.Compose([
    transforms.Resize((2316, 3088)),
    transforms.ToTensor(),
])

currentDir = os.getcwd()
trainFolder = currentDir + "\\resources\\trainData"
testFolder = currentDir + "\\resources\\testData"
validFolder = currentDir + "\\resources\\validData"

trainDataset =  FaceDataset(trainFolder, transform=transform)
testDataset = FaceDataset(testFolder, transform=transform)
validDataset = FaceDataset(validFolder, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)
validLoader = DataLoader(validDataset, batch_size=32, shuffle=False)

numEpochs = 10
trainLosses, validLosses = [],[]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FaceClassifier(num_classes=1)
model.to(device)
criterion = nn.CrossEntropyLoss()
# lr is learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(numEpochs):
    model.train()
    runningLoss = 0.0
    for images, labels in(trainLoader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item() * labels.size(0)
    trainLoss = runningLoss / len(trainLoader.dataset)
    trainLosses.append(trainLoss)

    model.eval()
    runningLoss = 0.0
    with torch.no_grad():
        for images, labels in validLoader():
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            runningLoss += loss.item() * labels.size(0)
    validLoss = runningLoss / len(validLoader.dataset)
    validLosses.append(validLoss)
    print(f"Epoch {epoch+1}/{numEpochs} - Train Loss: {trainLoss}, Validation Loss: {validLoss}")

dataset = FaceDataset(data_dir= os.getcwd() + "\\resources\\trainData")

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predicitons")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

cont = True
while cont:
    test_image = input("Put image Path or (n): ")
    if test_image.lower() == "n":
        break
    transform = transforms.Compose([
        transforms.Resize((2316, 3088)),
        transforms.ToTensor()
    ])

    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)

    class_names = dataset.classes
    visualize_predictions(original_image, probabilities, class_names)
    