## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    N_CLASSES = 68
    
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # the output_size of each feature map: (W-F)/S+1 = (244-5)/1+1 = 240
        # output tensor size: (32, 240, 240)
        # tensor size after pooling: (32, 120, 120)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers: with kernel_size=2, stride=2
        # output tensor size after pooling: (out_channel, img_size/2, img_size/2)
        self.pool = nn.MaxPool2d(2, 2) 
        
        # multiple conv layers: with kernel_size=5
        # the output size of each feature map: (W-F)/S+1 = (120-5)/1+1=116
        # output tensor size: (64, 116, 116)
        # tensor size after pooling: (64, 58, 58)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # fully-connected layers:
        # Input feature size: 64 * (58*58) (filtered then pooled map size)
        self.fc1 = nn.Linear(64*58*58, 136)
        
        # other layers (such as dropout or batch normalization) to avoid overfitting
        # Dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # Finally, create output channels for N_CLASSES
        self.fc2 = nn.Linear(136, Net.N_CLASSES)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # Passing the first conv layer + pool
        x = self.pool(F.relu(self.conv1(x)))
        # Passing the second conv layer + pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatenning before passing through the fully-connected layer
        x = x.view(x.size(0), -1)
        
        # Passing through 02 linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
