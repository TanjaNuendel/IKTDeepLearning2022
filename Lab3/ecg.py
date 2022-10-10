import torch
import numpy as np
import math
from tqdm import tqdm
#import pandas as pd
import random

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

dirname = os.path.dirname(__file__)
normal_directory = os.path.join(dirname, ".data", "normal")
abnormal_directory = os.path.join(dirname, ".data", "abnormal")

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
                # label 1 for "normal" data
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
                # label 0 for "abnormal" data 
                single_electrode_measurements.append(int(0))
                abnormal.append(single_electrode_measurements)

#print(normal.extend(abnormal))
all_data = normal + abnormal
all_data = np.asarray(all_data)
np.random.shuffle(all_data)
x_training = np.asarray(all_data)[:int(len(all_data)*split_ratio),0:75]
x_validation = np.asarray(all_data)[int(len(all_data)*split_ratio):,0:75]
y_training = np.asarray(all_data)[:int(len(all_data)*split_ratio),75]
y_validation = np.asarray(all_data)[int(len(all_data)*split_ratio):,75]


#print(y_validation)


#abnormal_training = np.asarray(abnormal)[:int(len(abnormal)*split_ratio),0:75]
#abnormal_validation = np.asarray(abnormal)[int(len(abnormal)*split_ratio):,75]

#training = normal_training.append(abnormal_training)
#validation = normal_validation.append(abnormal_validation)


class ECGDataset(Dataset):
    def __init__(self, mode):
        #data
        self.x = torch.from_numpy(x_training if mode=="training" else x_validation).to(torch.float32)
        #print(self.x.shape)
        #self.x = dataset[:, 0:7]
        #labels
        self.y = torch.from_numpy(y_training if mode=="training" else y_validation).to(torch.int64)
        #print(torch.from_numpy(x_training).to(torch.float32)[0])
        #print(self.y)
        #self.y = dataset[:, [7]]

        # number of samples
        self.n_samples = x_training.shape[0] if mode=="training" else x_validation.shape[0]

    def __getitem__(self, index):
        #print(self.x[index])
        return self.x[index], self.y[index].item()

    def __len__(self):
        return self.n_samples


py_ecgdata = ECGDataset("training")
py_ecgdata_validation = ECGDataset("val")
#print(py_ecgdata.__getitem__(0))

py_ecgloader = DataLoader(dataset=py_ecgdata, batch_size=32, shuffle=True, num_workers=0)
py_ecgloader_validation = DataLoader(dataset=py_ecgdata_validation, batch_size=32, shuffle=True, num_workers=0)

#dataiter = iter(py_ecgloader)
#images, labels = dataiter.next()
#print(labels)

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
hidden1_size = 256
hidden2_size = 64
learning_rate = 0.00001
num_epochs = 40

model = ECGNet(input_size, hidden1_size, hidden2_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#print(len(py_ecgloader))

def train():
    t_loss = []
    v_loss = []
    with tqdm(total=num_epochs) as pbar:
        for e in range(num_epochs):
            model.train()
            for i, (inputs,labels) in enumerate(py_ecgloader):       
                optimizer.zero_grad()
                
                scores = model(inputs)

                loss = criterion(scores,labels)
                #if(i%2000 == 0):
                #    print(loss.item())
                
                
                loss.backward()
                optimizer.step()

                #print(loss)
            t_loss.append(loss.item())
            pbar.update(1)
            model.eval()
            for i, (inputs,labels) in enumerate(py_ecgloader_validation):  
                scores = model(inputs)
                loss = criterion(scores,labels)
            v_loss.append(loss.item())
            
        

    print("training finished")

    import matplotlib.pyplot as plt
    # Plot training & validation accuracy values
    plt.plot(t_loss)
    plt.plot(v_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



x_validation = np.asarray(all_data)[int(len(all_data)*split_ratio):,0:75]
y_validation = np.asarray(all_data)[int(len(all_data)*split_ratio):,75]

class ECGTestDataset(Dataset):
    def __init__(self):
        #data
        self.x = torch.from_numpy(x_validation).to(torch.float32)
        #labels
        self.y = torch.from_numpy(y_validation).to(torch.int64)
        # number of samples
        self.n_samples = x_validation.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

py_ecgtestdata = ECGTestDataset()

#py_ecgtestloader = DataLoader(dataset=py_ecgtestdata, batch_size=4, shuffle=True, num_workers=0)


if __name__ == '__main__':
    print("main")
    train()
    #test()