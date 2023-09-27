"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        return 0*input[:,0]

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        self.hid3 = None
        return 0*input[:,0]

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()

    def forward(self, input):
        self.hid1 = None
        self.hid2 = None
        return 0*input[:,0]
