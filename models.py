import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.conv5 = nn.Conv2d(128, 256, 1)
        
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(2,2)
        self.pool5 = nn.MaxPool2d(2,2)
        
        self.dropout1 = nn.Dropout(p=0.0)
        self.dropout2 = nn.Dropout(p=0.0)
        self.dropout3 = nn.Dropout(p=0.0)
        self.dropout4 = nn.Dropout(p=0.0)
        self.dropout5 = nn.Dropout(p=0.0)
        self.dropout6 = nn.Dropout(p=0.6)
        self.dropout7 = nn.Dropout(p=0.6)
        
        self.fc1 = nn.Linear(9216, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 136)
              
    def forward(self, x):

        x = self.dropout1(self.pool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.elu(self.conv4(x))))
        x = self.dropout5(self.pool5(F.elu(self.conv5(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout6(F.elu(self.fc1(x)))
        x = self.dropout7(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
