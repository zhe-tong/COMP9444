#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        transform_data = transforms.Compose([

            transforms.Resize(240),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(180),
            transforms.RandomRotation(45),
            # transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])
        return transform_data
    elif mode == 'test':
        transform_data = transforms.Compose([

            transforms.Resize(240),
            transforms.RandomCrop(180),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])
        return transform_data


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4, padding=(1, 2))
        self.conv2 = nn.Conv2d(96, 256, (5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 64, (3, 3), stride=1, padding=1)
        self.pooling = nn.MaxPool2d((3, 3), stride=2)
        self.localResponseNorm = nn.LocalResponseNorm(5)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.linear3 = nn.Linear(2048, 8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.localResponseNorm(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.localResponseNorm(x)
        x = self.pooling(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x


net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

loss_func = torch.nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return


scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 100
epochs = 100
