import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import matplotlib.pyplot as plt
num_epochs = 30

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

batch_size = 64
test_batch_size = 64
input_size = 3072

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Conv7Net_NoRelu(nn.Module):
    def __init__(self):
        super(Conv7Net_NoRelu, self).__init__()
        #Layer 1 - Output dimension - 28*28*8
        # Conv2d (input channel, output channel, kernel size)
        self.conv1 = nn.Conv2d(3, 84, kernel_size=5, stride=2, padding=1)
        #self.pool1 = nn.MaxPool2d(2,2)
        
        #Layer 2  - Output dimension - 24*24*16
        self.conv2 = nn.Conv2d(84, 256, kernel_size=5, stride=2, padding=1)
        #self.pool2 = nn.MaxPool2d(3)

        #Layer 3  - Output dimension - 20*20*16
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        #self.pool3 = nn.MaxPool2d(3)
        
        #Layer 4  - Output dimension - 16*16*16
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        #self.pool4 = nn.MaxPool2d(3)
        
        #Layer 5, 6, and 7
        self.fc1 = nn.Linear(16*16*16,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
                
        #flatten the tensor for the FC
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #return F.log_softmax(x, dim=1)
        return x

net = Conv7Net_NoRelu()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# Construct our model by instantiating the class defined above
ConvModel2 = Conv7Net_NoRelu()
ConvModel2.to(device)
num_epochs = 100
#print(model)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ConvModel2.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)
#optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
#optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.0001, weight_decay=5e-4)


for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)

        #images = images.reshape(-1, 32*32*3)   
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        #print(images.shape)
        outputs = ConvModel2(images)

        loss = criterion(outputs, labels)    
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_batches += 1     
        batch_loss += loss.item()

    avg_loss_epoch = batch_loss/total_batches
    print ('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]' 
                   .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))

# Test the Model
correct = 0.
total = 0.
predicted = 0.
for images, labels in test_loader:
    #images = images.reshape(-1, 3*32*32)
    #print(labels)
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    outputs_Conv_test = ConvModel2(images)
    _, predicted = torch.max(outputs_Conv_test.data, 1)
    #print(predicted)
    total += labels.size(0) 
    
    correct += (predicted == labels).sum().item()
    
print('Accuracy of the network on the 10000 test images: %d %%' % (     100 * correct / total))
       
