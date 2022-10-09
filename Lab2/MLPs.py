import torch
import numpy as np
import math
#import pandas as pd

import os

import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#import tensorflow as tf
#import keras


os.environ['KMP_DUPLICATE_LIB_OK']='True'

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


#df = pd.read_csv("ecoli.data")
#print(df)

datalines = []
with open("/Users/jalell/Documents/coding/ikt450/task21/ecoli.data", "r") as f:
    d = f.readlines()
    for i in d:
        nospaces = [x for x in i.split(" ") if x]
        #print(nospaces)
        datalines.append([a.replace("\n", "") for a in nospaces])

dataset = []

for line in np.array(datalines):
    if(line[8] != "cp" and line[8] != "im"):
        break
    current = []
    for el in line:
        if(isfloat(el)):
            current.append(float(el))
        else: 
            if(el == "cp"):
                current.append(float(0))
            if(el == "im"):
                current.append(float(1))
    dataset.append(current)

#print(dataset)
dataset = np.array(dataset)
#print(dataset)

#filtered_dataset = []
#for el in dataset:
#    if el[8] == "cp" or el[8] == "im":
#        filtered_dataset.append(el)

#filtered_dataset = np.array(filtered_dataset)
#dataset = filtered_dataset

#print(dataset)
#print(dataset.astype(np.float))





##############################
# MLP from scratch only using python 


# 7 inputs for each ecoli value
i1=i2=i3=i4=i5=i6=i7 = 0
# 2 neurons in layer 1
n1=n2=0
# weights of first layer
w11 = 0.1
w12 = 0.3
w13 = 0.2
w14 = 0.7
w15 = 0.3
w16 = 0.1
w17 = 0.04
w21 = 0.6
w22 = 0.09
w23 = 0.1
w24 = 0.2
w25 = 0.8
w26 = 0.2
w27 = 0.5

# weights of second layer
w_11 = 0.8
w_12 = 0.4

# output
o = 0.0

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def MLP(line):
    #set inputs
    i1 = float(line[1])
    i2 = float(line[2])
    i3 = float(line[3])
    i4 = float(line[4])
    i5 = float(line[5])
    i6 = float(line[6])
    i7 = float(line[7])

    n1 = sigmoid(w11*i1 + w12*i2 + w13*i3 + w14*i4 + w15*i5 + w16*i6 + w17*i7)
    n2 = sigmoid(w21*i1 + w22*i2 + w23*i3 + w24*i4 + w25*i5 + w26*i6 + w27*i7)

    o = sigmoid(w_11*n1 + w_12*n2)

    return o

#for i in range(dataset.shape[0]-1):
    #print(MLP(dataset[i]))

#print(MLP(dataset[204]))
#print(dataset.shape)


#print(torch.__version__)
#print(keras.__version__)

#print(torch.from_numpy(dataset[:, [7]]))
#print(dataset.shape[0])


####################################
# MLP with PyTorch

#print(type(dataset[1][2]))

class EcoliDataset(Dataset):
    def __init__(self):
        #data
        self.x = torch.from_numpy(dataset[:, 0:7]).to(torch.float32)
        #self.x = dataset[:, 0:7]
        #labels
        self.y = torch.from_numpy(dataset[:, [7]]).to(torch.float32)
        #self.y = dataset[:, [7]]

        # number of samples
        self.n_samples = dataset.shape[0]

    def __getitem__(self, index):
        #print(self.x[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


py_ecolidata = EcoliDataset()
#print(py_ecolidata.x)
#print(py_ecolidata.y)
py_ecoliloader = DataLoader(dataset=py_ecolidata, batch_size=4, shuffle=True, num_workers=0)


class EcoliNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(EcoliNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        #print(x)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

input_size = 7
num_classes = 2
hidden1_size = 100
hidden2_size = 50
learning_rate = 0.001
batch_size = 64
num_epochs = 1

model = EcoliNet(input_size, hidden1_size, hidden2_size, num_classes)
#model.to(torch.float)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for i, (inputs,labels) in enumerate(py_ecoliloader):
        #print(inputs)
        #print(labels)
        
        scores = model(inputs)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        #loss.backward()

        #optimizer.step()