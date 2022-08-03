import torch.nn as nn
import torch.nn.functional as F
import torch


class Fusion_Outputs(nn.Module):
    def __init__(self,  filter1, filter2, kernel):
        super(Fusion_Outputs, self).__init__()

        self.input1_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input1_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)
        self.input1_fc = nn.Linear(6*36*36, 10)

        self.input2_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input2_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)
        self.input2_fc = nn.Linear(6*36*36, 10)

        self.input3_conv1 = nn.Conv2d(1, filter1, kernel_size=kernel, stride=1)
        self.input3_conv2 = nn.Conv2d(filter1, filter2, kernel_size=kernel, stride=1)
        self.input3_fc = nn.Linear(6*36*36, 10)

    def forward(self, x1, x2, x3):

        x1 = F.relu(self.input1_conv1(x1))
        x1 = F.relu(self.input1_conv2(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.relu(self.input1_fc(x1))
        x1 = F.softmax(x1, dim=1)

        x2 = F.relu(self.input2_conv1(x2))
        x2 = F.relu(self.input2_conv2(x2))
        x2 = torch.flatten(x2, 1)
        x2 = F.relu(self.input2_fc(x2))
        x2 = F.softmax(x2, dim=1)

        x3 = F.relu(self.input3_conv1(x3))
        x3 = F.relu(self.input3_conv2(x3))
        x3 = torch.flatten(x3, 1)
        x3 = F.relu(self.input1_fc(x3))
        x3 = F.softmax(x3, dim=1)

        out = x1 + x2 + x3
        return out


model = Fusion_Outputs(filter1=6, filter2=6, kernel=3)

# to train or test
y_pred = model(x1, x2, x3)