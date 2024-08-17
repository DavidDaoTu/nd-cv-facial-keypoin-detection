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
        # the output_size of each feature map: (W-F)/S+1 = (224-5)/1+1 = 220
        # output tensor size: (32, 220, 220)
        # tensor size after pooling: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5) 
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers: with kernel_size=2, stride=2
        # output tensor size after pooling: (out_channel, img_size/2, img_size/2)
        self.pool = nn.MaxPool2d(2, 2) 
        
        # multiple conv layers: 
        # Convolutional layer 2: with kernel_size=5
        # the output size of each feature map: (W-F)/S+1 = (110-5)/1+1=106
        # output tensor size: (64, 106, 106)
        # tensor size after pooling: (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # Convolutional layer 3: with kernel_size=3
        # the output size of each feature map: (W-F)/S+1 = (53-3)/1+1=51
        # output tensor size: (128, 51, 51)
        # tensor size after pooling: (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Convolutional layer 4: with kernel_size=3
        # the output size of each feature map: (W-F)/S+1 = (25-3)/1+1=23
        # output tensor size: (256, 23, 23)
        # tensor size after pooling: (256, 11, 11)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # Convolutional layer 5: with kernel_size=3
        # the output size of each feature map: (W-F)/S+1 = (11-3)/1+1=9
        # output tensor size: (512, 9, 9)
        # tensor size after pooling: (512, 4, 4)
        self.conv5 = nn.Conv2d(256, 512, 3)
        
        # fully-connected layers:
        # Input feature size: 512 * (4*4) (filtered then pooled map size)
        self.fc1 = nn.Linear(512*4*4, 2048)
        
        # other layers (such as dropout or batch normalization) to avoid overfitting
        # Dropout with p=0.21
        self.fc_drop = nn.Dropout(p=0.21)
        
        # other linear class
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        
        # Finally, create output channels for N_CLASSES
        self.fc5 = nn.Linear(256, Net.N_CLASSES * 2)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # Passing the first conv layer + pool
        x = self.pool(F.relu(self.conv1(x)))
        # Passing the second conv layer + pool
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flatenning before passing through the fully-connected layer
        x = x.view(x.size(0), -1)
        
        # Passing through 03 linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc_drop(x)
        
        x = F.relu(self.fc3(x))
        x = self.fc_drop(x)
        
        x = F.relu(self.fc4(x))
        x = self.fc_drop(x)
        
        x = self.fc5(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
