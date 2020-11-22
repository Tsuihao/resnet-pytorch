import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
#from torchvision import dataset, models, transforms

# check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class residual_block(nn.Module):
    def __init__(self, in_channels, intermed_channels, identity_downsample=None, stride=1):
        super(residual_block, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, intermed_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermed_channels)
        self.conv2 = nn.Conv2d(intermed_channels, intermed_channels, kernel_size=3, stride=stride, padding=1) # the stride here is very tricky
        self.bn2 = nn.BatchNorm2d(intermed_channels)
        self.conv3 = nn.Conv2d(intermed_channels, intermed_channels*self.expansion,  kernel_size=1, stride=1, padding=0)       
        self.bn3 = nn.BatchNorm2d(intermed_channels*self.expansion)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone() 
        
        # the order is conv -> batch norm -> relu 
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) # no relu in the last layer

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity) # nn.Sequential, make sure same dimension
        
        out += identity # shortcut connection (element-wise addtion)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, residual_block, layers, in_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        
        self.stage2 = self._make_stage(residual_block, layers[0], 64, stride=1) # Resnet paper stage 2 has no downsample 
        self.stage3 = self._make_stage(residual_block, layers[1], 128, stride=2)
        self.stage4 = self._make_stage(residual_block, layers[2], 256, stride=2)
        self.stage5 = self._make_stage(residual_block, layers[3], 512, stride=2) 
                
        self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        
    def _make_stage(self, residual_block, num_residual_block, in_channels, stride):
        identity_downsample = None
        layers = []
        
        # The most confusing part
        if stride != 1 or self.in_channels != in_channels * 4:
             identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, in_channels*4, kernel_size=1, stride=stride),
                                                 nn.BatchNorm2d(in_channels*4))
        

        layers.append(residual_block(self.in_channels, in_channels, identity_downsample, stride))
        self.in_channels = in_channels * 4

        for i in range(num_residual_block-1):
            layers.append(residual_block(self.in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        # Stage 1
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        # Stage 2-5
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        
        # average pool
        out = self.avepool(out)  # (7,7,2048) -> (1,1,2048)
        out = out.reshape(out.shape[0], -1)
        # fc
        out = self.fc(out)
        return out
        

ResNet50 = ResNet(residual_block, [3,4,6,3], 3, 1000)
x = torch.randn(4, 3, 224, 224)
y = ResNet50(x).to(device)

