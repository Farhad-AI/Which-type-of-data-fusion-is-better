import torch.nn as nn
import torch.nn.functional as F
import torch


class Fusion_In_Input(nn.Module):
    def __init__(self,  filter1, filter2, kernel):
        super(Fusion_In_Input, self).__init__()

        self.conv1 = nn.Conv2d(3, filter1, kernel_size=kernel, stride=1)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)

        self.fully_connected = nn.Linear(6*36*36, 10)

    def forward(self, x1, x2, x3):
        x_new = torch.concat((x1, x2, x3), 1)

        x_new = F.relu(self.conv1(x_new))
        x_new = F.relu(self.conv2(x_new))

        x_new = torch.flatten(x_new, 1)
        out = F.softmax(self.fully_connected(x_new))
        return out


model = Fusion_In_Input(filter1=6, filter2=6, kernel=3)

# to train or test
y_pred = model(x1, x2, x3)
