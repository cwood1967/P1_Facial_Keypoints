## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.drop2d = nn.Dropout2d(.25)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.drop = nn.Dropout(.25)
        self.fc1 = nn.Linear(12*12*256, 10000)
        self.fc2 = nn.Linear(10000, 4000)
        self.fc3 = nn.Linear(4000, 500)
        self.fc4 = nn.Linear(500, 2*68)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
   
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        activation = F.relu
        x = self.conv1(x)
        x = activation(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = activation(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = activation(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = activation(x)
        x = self.pool(x)
        x = self.drop2d(x)
        
        x = x.view(-1, self.numflat(x))
        x = self.fc1(x)
        x = activation(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = activation(x)
        x = self.fc3(x)
        x = activation(x)
        x = self.fc4(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def numflat(self, x):
        size = x.size()[1:]
        numf = 1
        for s in size:
            numf *= s

        return numf