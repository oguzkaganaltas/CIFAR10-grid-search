import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the ANN
class MyModel(nn.Module):

    def __init__(self, num_layer1, num_layer2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32*1, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # not have to add softmax layer here
        return x