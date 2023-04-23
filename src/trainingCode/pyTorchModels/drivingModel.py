import torch.nn as nn

roadWidth = 180
roadHeight = 56

class netNvidia(nn.Module):
    def __init__(self):
        super(netNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,24,5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24,36,5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36,48, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48,64,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding=1),
            #nn.Flatten(),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*4*19, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, input):
        input = input.view(input.size(0), 3, roadHeight, roadWidth)
        output = self.conv_layers(input)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

driverModel = netNvidia()