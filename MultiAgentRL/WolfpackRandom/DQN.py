import torch.nn.functional as F
import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, expDepth=8):
        super(DQN, self).__init__()
        self.expDepth = expDepth
        self.conv1 = nn.Conv2d(self.expDepth, 16, kernel_size=8, stride=2).double()
        self.bn1 = nn.BatchNorm2d(16).double()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2).double()
        self.bn2 = nn.BatchNorm2d(32).double()
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1).double()
        self.bn3 = nn.BatchNorm2d(16).double()
        self.head = nn.Linear(240, 6).double()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        