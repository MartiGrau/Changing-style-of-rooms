from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import copy
import time
import xlwt
from xlwt import Workbook

plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/imatge/mgrau/work/RESULTATS_FINALS/classifier/3'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=150,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_clases = 3

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
#Generate Excel template with the real and prediction labels
#---------------------------------------------

def excel_template(inputs, true,preds, percentages):
    wb = Workbook()
    # add_sheet is used to create sheet
    sheet = wb.add_sheet('classificator')
    sheet.write(0, 0, 'Img Name')
    sheet.write(0, 1, 'True Style')
    sheet.write(0, 2, 'Predicted Style')
    sheet.write(0, 3, 'Rustic %')
    sheet.write(0, 4, 'Minimalist %')
    sheet.write(0, 5, 'Clasic %')

    for x in range(inputs.size()[0]):
        sheet.write(x+1, 0, 'Img{}'.format(x+1))
        sheet.write(x+1, 1, true[x])
        sheet.write(x+1, 2, preds[x])
        sheet.write(x+1, 3, percentages[x][2].item())
        sheet.write(x+1, 4, percentages[x][1].item())
        sheet.write(x+1, 5, percentages[x][0].item())

    return wb


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probability = torch.nn.functional.softmax(outputs)

#!!!!EL CALCUL D'AIXO INFLUEIX SEGONS EL NUM D'IMATGES A EVALUAR (batch_size=507) LINEA 40 !!!##
#---------------------------------------------------------------------
            #COFUSION MATRIX
            y_true = list()
            y_pred = list()

            for j in range(inputs.size()[0]):
                y_true.append(class_names[labels[j]])
                y_pred.append(class_names[preds[j]])

            cf=confusion_matrix(y_true, y_pred, labels=["clasic", "minimalist", "rustic"])
            print ("CONFUSION MATRIX")
            print (pd.DataFrame(cf, index=['true: clasic', 'true:minim', 'true:rustic'], columns=['pred: clasic', 'pred:minim', 'pred:rustic']))
#----------------------------------------------------------------
            #Create Excel Template
            wb = excel_template(inputs, y_true, y_pred, probability)
            wb.save('/imatge/mgrau/work/RESULTATS_FINALS/classification/final/{}_3.xls'.format(j))

model = torch.load('/imatge/mgrau/PycharmProjects/classificator/2nd_mymodel_conv.pt')
model.eval()

visualize_model(model)

