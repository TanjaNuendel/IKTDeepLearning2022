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
        print(self.x[index])
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
        print(x)
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

criterion = nn.L1Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for i, (inputs,labels) in enumerate(py_ecoliloader):
        #print(inputs)
        #print(labels)
        
        scores = model(inputs)

        #loss = criterion(scores,labels)