from pickletools import optimize
from wsgiref import validate
from xml.dom import ValidationErr
import torch
import numpy as np
import math
#import pandas as pd

import tarfile

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

normal_directory = "/Users/jalell/Documents/coding/ikt450/task2_3/ecg/normal"
abnormal_directory = "/Users/jalell/Documents/coding/ikt450/task2_3/ecg/abnormal"

normal = []
abnormal = []

split_ratio = 0.8

for filename in os.listdir(normal_directory):
    f = os.path.join(normal_directory, filename)
    if os.path.isfile(f):
        name = f.split("/")[-1]
        if(name.split(".")[1] != "ann"):
            with open(normal_directory +"/" + name) as datafile:
                single_electrode_measurements = []
                input_size = len(datafile.readlines())
                padding = [0] * (75 - input_size if input_size < 75 else 0)
                datafile.seek(0)
                line_counter = 0
                for line in datafile:
                    if(line_counter < 75):
                        single_electrode_measurements.append(float(line.split(" ")[-1].replace("\n", "")))
                        #print(line_counter)
                    line_counter += 1
                single_electrode_measurements.extend(padding)
                single_electrode_measurements.append(int(1))
                normal.append(single_electrode_measurements)
                #print(len(single_electrode_measurements))

#print(len(normal[0]))

for filename in os.listdir(abnormal_directory):
    f = os.path.join(abnormal_directory, filename)
    if os.path.isfile(f):
        name = f.split("/")[-1]
        if(name.split(".")[1] != "ann"):
            with open(abnormal_directory + "/" + name) as datafile:
                single_electrode_measurements = []
                input_size = len(datafile.readlines())
                padding = [0] * (75 - input_size if input_size < 75 else 0)
                datafile.seek(0)
                line_counter = 0
                for line in datafile:
                    if(line_counter < 75):
                        single_electrode_measurements.append(float(line.split(" ")[-1].replace("\n", "")))
                    line_counter += 1
                single_electrode_measurements.extend(padding)
                single_electrode_measurements.append(int(0))
                abnormal.append(single_electrode_measurements)

#print(len(abnormal[0]))

#print(normal.extend(abnormal))
all_data = normal + abnormal
#print(np.asarray(all_data)[:,75])
training = np.asarray(all_data)[:int(len(all_data)*split_ratio),0:75]
#validation = np.asarray(all_data)[int(len(all_data)*split_ratio):,75]
validation = np.asarray(all_data)[:int(len(all_data)*split_ratio),75]

#abnormal_training = np.asarray(abnormal)[:int(len(abnormal)*split_ratio),0:75]
#abnormal_validation = np.asarray(abnormal)[int(len(abnormal)*split_ratio):,75]

#training = normal_training.append(abnormal_training)
#validation = normal_validation.append(abnormal_validation)


class ECGDataset(Dataset):
    def __init__(self):
        #data
        self.x = torch.from_numpy(training).to(torch.float32)
        #print(self.x.shape)
        #self.x = dataset[:, 0:7]
        #labels
        self.y = torch.from_numpy(validation).to(torch.int64)
        #print(self.y)
        #self.y = dataset[:, [7]]

        # number of samples
        self.n_samples = training.shape[0]

    def __getitem__(self, index):
        #print(self.x[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


py_ecgdata = ECGDataset()


py_ecgloader = DataLoader(dataset=py_ecgdata, batch_size=4, shuffle=True, num_workers=0)


class ECGNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(ECGNet, self).__init__()
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

input_size = 75
num_classes = 2
hidden1_size = 100
hidden2_size = 50
learning_rate = 0.001
batch_size = 64
num_epochs = 100

model = ECGNet(input_size, hidden1_size, hidden2_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for i, (inputs,labels) in enumerate(py_ecgloader):       
        scores = model(inputs)

        loss = criterion(scores,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
