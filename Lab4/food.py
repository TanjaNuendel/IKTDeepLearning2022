import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import natsort

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())

main_dir = "/Users/jalell/Documents/coding/ikt450/task24/Food-11/"

dirname = os.path.dirname(__file__)
main_dir = os.path.join(dirname, ".data", "Food-11")
#print(data_path)



# Hyper-parameters 
num_epochs = 2
batch_size = 4
learning_rate = 0.0001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Resize((32,32)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class FoodDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.images = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.images[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = int(self.images[idx].split("_")[0])
        return tensor_image, label

train_dataset = FoodDataset(os.path.join(main_dir, "training"), transform)
food_validation_data = FoodDataset(os.path.join(main_dir, "validation"), transform)
food_evaluation_data = FoodDataset(os.path.join(main_dir, "evaluation"), transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(food_validation_data, batch_size=batch_size, num_workers=0, shuffle=True)


classes = ('bread', 'dairy product', 'dessert', 'egg',
           'fried food', 'meat', 'pasta', 'rice', 'seafood', 'soup', 'vegetable/fruit')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




'''
conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)

print(images.shape)
x = conv1(images)
print(x.shape)
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)

'''

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
#print(images)
#print(labels)
# show images
#imshow(torchvision.utils.make_grid(images))
#print(train_dataset.__getitem__(1))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # for pooling 10,10 -> 16*4*4 (512 image size)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,11)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        #print(x.shape)
        x = x.view(-1, 16 * 5 * 5)            # -> n, 250.000
        #print(x)
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 11
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#print(len(train_loader))

def train():
    n_total_steps = len(train_loader)
    with tqdm(total=n_total_steps) as pbar:
        for epoch in range(num_epochs):        
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                #print("training")
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                # input_layer: 3 input channels, 6 output channels, 5 kernel size
                images = images.to(device)
                labels = labels.to(device)
                #print(labels)
                # Forward pass
                outputs = model(images)
                #print(outputs)

                loss = criterion(outputs, labels)

                # Backward and optimize
                
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            #pbar.update(1)

    print('Finished Training')
        
def test():
    t_loss = []
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(11)]
        n_class_samples = [0 for i in range(11)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            for i in range(batch_size):
                if(labels.size(dim=0) == 4):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')

if __name__ == '__main__':
    print("main")
    train()
    #test()