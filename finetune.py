#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
from os import listdir
import time
import torch
import pandas as pd
import csv
import pickle
import copy
import torchvision
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch import nn
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# In[2]:


trainfile = pd.read_csv('./train.csv')
testfile = pd.read_csv('./test.csv')


# In[103]:


test_id = []
for i in testfile['id']:
    test_id.append(i)
# print(test_id)


# In[4]:


train_data = {}

with open('./train.csv') as infile:
    reader = csv.DictReader(infile)           # read rows into a dictionary format
    for row in reader:
        for header, value in row.items():
          try:
            train_data[header].append(value)
          except KeyError:
            train_data[header] = [value]
id_train = train_data['id']
landmark_id = train_data['landmark_id']
dictionary = dict(zip(id_train, landmark_id))
id_train_jpg = []
for i in id_train:
    id_train_jpg.append(i+".jpg")
# print(id_train)


# In[5]:


imagefiles = [i for i in listdir('./images')]
image_id_jpg = []
image_label=[]
image_id = []
for i in imagefiles:
    image_id_jpg.append(i)
    fn = i.replace('.jpg','')
    if fn in dictionary:
        image_id.append(fn)
        image_label.append(dictionary[fn])
image_dictionary = dict(zip(image_id, image_label))
bb = []
for i in image_label:
    bb.append(int(i))
k = np.array(bb)
# print(k)
cc=[]
for i in landmark_id:
    cc.append(int(i))
k1 = np.array(cc)
# print(k1)


# In[6]:


x=np.array(imagefiles).reshape(1,-1).transpose()
y=np.array(image_label)
# print(x)
# print(y)


# In[7]:


class GetDataset(Dataset):

    def __init__(self, root_dir, data, transform=None):
        self.x = pd.read_csv(root_dir)
        self.path = data
        self.root_dir = root_dir
        self.transform = transform
        self.image_id = data
        self.y = k1

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        s = self.path, id_train_jpg[idx]
        img_name = os.path.join(*s)
        image = io.imread(img_name)
        img = transforms.ToPILImage()(image)
        label = self.y[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# In[8]:


data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# In[9]:


dataset_train = GetDataset(root_dir='./train.csv', data = "./images", transform = data_transforms)


# In[10]:


dataset_sizes = len(dataset_train)


# In[11]:


class_names = ['St.Stephan’sCathedral,Austria', 'Teide,Spain', 'Tallinn,Estonia', 'Brugge,Belgium', 'Montreal,Canada', 'ItsukushimaShrine,Japan', 'Shanghai,China', 'Brisbane,Australia', 'Edinburgh,Scotland', 'Stockholm,Sweden']


# In[12]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[13]:


dataloaders = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)


# In[14]:


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

inputs, classes = next(iter(dataloaders))
# print(inputs, classes)

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


# In[15]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders:   #for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):    # with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
#                 print(outputs)
                _, preds = torch.max(outputs, 1)   # maximum number for each row
                loss = criterion(outputs, labels)
        
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes      # train.csv?
        epoch_acc = running_corrects.double() / dataset_sizes     #dataset_sizes

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


# In[16]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)


# In[17]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)


# In[ ]:


torch.save(model_ft, 'model.pt')


# In[ ]:


model_ft = torch.load('model.pt')


# In[81]:


def prediction_model(data_x, model):
    prediction = []
    for i in data_x:
        model.eval()
        log_py = model(i)
        pred = np.argmax(log_py.data.numpy(), axis=1) 
        prediction.append(pred.tolist())
    return(prediction)


# In[105]:


id_test_jpg = []
for i in test_id:
    id_test_jpg.append(i+".jpg")
# print(id_test_jpg)


# In[106]:


class GetDataset1(Dataset):

    def __init__(self, root_dir, data, transform=None):
        self.x = pd.read_csv(root_dir)
        self.path = data
        self.root_dir = root_dir
        self.transform = transform
        self.image_id = data
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        s = self.path, id_test_jpg[idx]
        img_name = os.path.join(*s)
        image = io.imread(img_name)
        img = transforms.ToPILImage()(image)
        if self.transform is not None:
            img = self.transform(img)

        return img


# In[107]:


dataset_test = GetDataset1(root_dir='./test.csv', data = "./images", transform = data_transforms)


# In[108]:


dataloaders1 = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4)


# In[109]:


predic_result = prediction_model(dataloaders1, model_ft)


# In[110]:


# print(predic_result)


# In[111]:


file = open('./submission.txt','w') 
file.write('landmark_id\n') 
for i in predic_result:
    for k in i:
        file.write(str(k))
        file.write("\n")
file.close()

