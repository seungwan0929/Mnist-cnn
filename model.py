import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module): 
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  
        self.fc1 = nn.Linear(64 * 12 * 12, 128)        
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))                
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  
        x = x.view(x.size(0), -1)               
        x = F.relu(self.fc1(x))
        return self.fc2(x)