import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

import torch.nn as nn

from torchvision.models import ResNet50_Weights

import torch.optim as optim
from torch.optim import lr_scheduler

def load_model():
    img_size = 224 #INCREASING THIS WILL MAKE VRAM USAGE EXPLODE
    test_batch_size=1

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #Don't mess with the normalize values. resnet prefers these values
        transforms.Resize(img_size, antialias=True), #Set image size to this
        transforms.CenterCrop(img_size) #Ensure image is a square input
        ])

    # Data loaders
    test_set = torchvision.datasets.Food101(root='./data',
                                            split ='test',
                                            transform=test_transforms,
                                            download=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=test_batch_size,
                                            shuffle=True,
                                            num_workers=2)
    
    return test_loader

def import_classes():
    classes_path = './food-101/meta/classes.txt'
    class_file = open(classes_path,'r')
    class_data = class_file.read()
    classes = class_data.split("\n")
    class_file.close()
    return classes

def load_model(model_path):
    checkpoint = torch.load(model_path)

    #Set up base model
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    #Change outputs to 102 possible outputs
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 102)

    if torch.cuda.is_available():
        model.cuda()

    #Set to True for training, set to false for testing
    for param in model.parameters():
        param.requires_grad=False

    #Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
