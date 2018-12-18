import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

def train(data_dir = 'flower_data', model=None):
    
    train_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    test_transforms = train_transforms = transforms.Compose([
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    if not model:
        print("1: vgg16, 2: vgg19")
        inp = input("Which Model do you want?")
        if inp == '1':
            print("VGG16 selected")
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

            
            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, 4096)),
                                    ('relu', nn.ReLU()),
                                    ('drop1', nn.Dropout(p=0.3)),
                                    ('fc2', nn.Linear(4096, 1000)),
                                    ('relu2', nn.ReLU()),
                                    ('drop2', nn.Dropout(p=0.3)),
                                    ('fc3', nn.Linear(1000, 512)),
                                    ('relu3', nn.ReLU()),
                                    ('drop3', nn.Dropout(p=0.3)),
                                    ('fc4', nn.Linear(512, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))    
            model.classifier = classifier

        elif inp=='2':
            print("vgg19 selected")
            model = models.vgg19(pretrained=True)
            
            for param in model.parameters():
                param.requires_grad = False

            classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(25088, 4096)),
                                    ('relu', nn.ReLU()),
                                    ('drop1', nn.Dropout(p=0.3)),
                                    ('fc2', nn.Linear(4096, 1000)),
                                    ('relu2', nn.ReLU()),
                                    ('drop2', nn.Dropout(p=0.3)),
                                    ('fc3', nn.Linear(1000, 512)),
                                    ('relu3', nn.ReLU()),
                                    ('drop3', nn.Dropout(p=0.3)),
                                    ('fc4', nn.Linear(512, 102)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ])) 
            model.classifier = classifier
        else:
            exit()


    
    criterion = nn.NLLLoss()
    epochs = int(input("How many Epochs?"))
    lr = float(input("How much learning rate?"))
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    device = input("'cpu' or 'cuda'?")

    model = model.to(device)
    device = torch.device(device)
    
    
    steps = 0
    print_every = 20

    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps%print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:

                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)                
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(validloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    print("------------------------")
    
    accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            ps = torch.exp(log_ps)
            
            top_p, top_class = ps.topk(1, dim=1)
            
            equals = top_class == labels.view(*top_class.shape)
            accuracy += sum(equals).item()
            total += len(labels)

    print("Accuracy is : ", accuracy/total)

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'model' : model,
        'state_dict' : model.state_dict(),
        'class_to_idx':model.class_to_idx
    }
    torch.save(checkpoint, 'checkpoint'+str(inp)+'.pth')
    
def load_checkpoint(file):
    checkpoint = torch.load(file)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


if __name__=="__main__":
    # model = load_checkpoint('checkpoint.pth')
    train()