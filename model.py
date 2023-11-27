import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.wake = True

    def forward(self, x):
        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        ent1 = x
        x = self.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        ent2 = x
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        ent3 = x
        x = self.dropout(x)
        x = self.fc2(x)
        return x.softmax(dim=1), ent1.softmax(dim=1), ent2.softmax(dim=1), ent3.softmax(dim=1)

class CifarNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.3)
        self.wake = True

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        ent1 = x
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        ent2 = x
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        ent3 = x
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.softmax(dim=1), ent1.softmax(dim=1), ent2.softmax(dim=1), ent3.softmax(dim=1)

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet_18(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self.wake = True

    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ent1 = x
        
        x = self.layer1(x)
        ent2 = x
        x = self.layer2(x)
        ent3 = x
        x = self.layer3(x)
        ent4 = x
        x = self.layer4(x)
        ent5 = x
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x.softmax(dim=1), ent1.softmax(dim=1), ent2.softmax(dim=1), ent3.softmax(dim=1), ent4.softmax(dim=1), ent5.softmax(dim=1)
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )


def build_model(config):
    """Builds a model based on the config"""
    model = ResNet_18(3, 10)#CifarNET()
    return model