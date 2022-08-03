import torch.nn as nn
import torch.nn.functional as F
import torch


class Fusion_Before_FC(nn.Module):
    def __init__(self,  filter1, filter2, kernel):
        super(Fusion_Before_FC, self).__init__()

        self.input1_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input1_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)

        self.input2_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input2_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)

        self.input3_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input3_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)

        self.fully_connected = nn.Linear(18*36*36, 10)

    def forward(self, x1, x2, x3):

        x1 = F.relu(self.input1_conv1(x1))
        x1 = F.relu(self.input1_conv2(x1))

        x2 = F.relu(self.input2_conv1(x2))
        x2 = F.relu(self.input2_conv2(x2))

        x3 = F.relu(self.input3_conv1(x3))
        x3 = F.relu(self.input3_conv2(x3))

        x_new = torch.concat((x1, x2, x3), 1)
        x_new = torch.flatten(x_new, 1)
        x_new = F.relu(self.fully_connected(x_new))
        out = F.softmax(x_new, dim=1)
        return out


model = Fusion_Before_FC(filter1=6, filter2=6, kernel=3)

# to train or test
y_pred = model(x1, x2, x3)