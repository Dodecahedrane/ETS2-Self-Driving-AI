import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation


class netWheel(nn.Module):
    def __init__(self):
        super(netWheel, self).__init__()
        self.fc = nn.Linear(512, 128)
        
        self.branch_a1 = nn.Linear(128, 32)
        self.branch_a2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        a = F.leaky_relu(self.branch_a1(x))
        out1 = self.branch_a2(a)
        return out1
    
resnet18 = torchvision.models.resnet18()
resnet18.fc = nn.Identity()
net_add=netWheel()
steerAngleModel = nn.Sequential(resnet18, net_add)