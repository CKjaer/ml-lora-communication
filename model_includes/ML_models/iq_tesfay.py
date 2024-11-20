import torch.nn as nn
import torch.nn.functional as F


class iq_tesfay(nn.Module):
    def __init__(self, M):
        super(iq_tesfay, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=M // 4, kernel_size=4, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=M // 4, out_channels=M // 2, kernel_size=4, stride=1, padding=2
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(M // 4)
        self.batchNorm2 = nn.BatchNorm2d(M // 2)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(M // 2 * (M // 4) * (M // 4), 4 * M)
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, 128)  # Output size is 128 for 128 labels
        self.batchNorm3 = nn.BatchNorm1d(4 * M)
        self.batchNorm4 = nn.BatchNorm1d(2 * M)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)

        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))

        # Fully connected layers
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
