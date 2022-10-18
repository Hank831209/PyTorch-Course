import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_dir = "./hymenoptera_data"
model_name = "resnet"
num_classes = 2
batch_size = 32
num_epochs = 15
feature_extract = True
input_size = 224

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
# image_datasets['train']: train dataset, image_datasets['val']: val dataset
dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ["train", "val"]}
# dataloaders_dict['train']: train Dataloader, dataloaders_dict['val']: val Dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():  # 
    param.requires_grad = False
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
print(model.layer1[0].conv1.weight.requires_grad)  # False
model = model.to(device)

# optimizer
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


