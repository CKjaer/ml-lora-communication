import torch.nn as nn
import torch.nn.functional as F

class IQ_cnn(nn.Module):
    def __init__(self, M):
        super(IQ_cnn, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//2, kernel_size=(1,7), stride=1, padding=(1,3))
        self.conv2 = nn.Conv2d(in_channels=M//2, out_channels=M, kernel_size=(1,7), stride=1, padding=(1,3))
        self.conv3 = nn.Conv2d(in_channels=M, out_channels=M*2, kernel_size=(1,7), stride=1, padding=(1,3))
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        
        # Dropout layer
        self.drop=nn.Dropout(0.1)

        self.batchNorm1=nn.BatchNorm2d(M//2)
        self.batchNorm2=nn.BatchNorm2d(M)
        self.batchNorm3=nn.BatchNorm2d(M*2)

        # Fully connected layers
        self.fc1 = nn.Linear(M*(M//2), M)  
        # self.fc2 = nn.Linear(4 * M, 2 * M)
        # self.fc3 = nn.Linear(2 * M, M)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.batchNorm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.batchNorm2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.batchNorm3(x)
        x = self.pool(x)
        x = self.drop(x)
        
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features