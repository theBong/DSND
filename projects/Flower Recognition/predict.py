import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image

def load_checkpoint(file):
    checkpoint = torch.load(file)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    


def predict():
    checkpointpath = input("Checkpoint path (relative to script or absolute) ")
    image_path = input("Imagepath on which prediction has to be made ")
    jsonpath = input("Enter json path: ")
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    device = input("'cpu' or 'cuda' ")
    topk = int(input("Enter the number of topk classes you want to see "))

    
    model = load_checkpoint(checkpointpath)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    model.to(device)
    image = process_image(image_path)   
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        image.unsqueeze_(0)
        image = image.float()
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(topk)

    top_names = []

    for c in top_class[0]:
        top_names += [cat_to_name[idx_to_class[c.item()]]]

    inp = input("1: To see top prediction, 2: To see topk predictions, 3: To see class names")
    if inp == '1':
        print(top_class[0][0].item())

    elif inp == '2':
        print(top_p[0])
        print(top_class[0])

    elif inp == '3':
        print(top_names)

    return top_p, top_class

def process_image(image): 
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array '''
    pil_image = Image.open(image)
    w, h =pil_image.size
    ratio = h/w
    if ratio > 1:
        pil_image = pil_image.resize((256, int(256/ratio)))
    elif ratio < 1:
        pil_image = pil_image.resize((int(256/ratio), 256))
    else:
        pil_image = pil_image.resize((256,256))

    size = pil_image.size
    pil_image = pil_image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112))

    np_image = np.array(pil_image)
    np_image = np_image/255
    means = ([0.485, 0.456, 0.406])
    stdevs = ([0.229, 0.224, 0.225])
    np_image = (np_image - means)/stdevs
    np_image = np_image.transpose(2,0,1)
    
    img_tensor = torch.from_numpy(np_image)
    return img_tensor

if __name__ == '__main__':
    predict()