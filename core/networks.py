
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class Classifier(nn.Module):
    def __init__(self, model_name, num_classes_list=[10], pretrained=True):
        super().__init__()

        self.model = eval(model_name)(pretrained=pretrained)

        if 'resnet' in model_name:
            self.in_channels = self.model.fc.in_features
            self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        self.flatten = nn.Flatten()
        self.mlp_list = nn.ModuleList([self.build_MLP(self.in_channels, num_classes) for num_classes in num_classes_list])

        self.initialize()
    
    def forward(self, xs):
        features = self.features(xs)
        features = self.flatten(features)
        return [mlp(features) for mlp in self.mlp_list]

    def build_MLP(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Linear(in_channels // 2, in_channels // 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            
            nn.Linear(in_channels // 4, num_classes)
        )

    def initialize(self):
        for m in self.mlp_list:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

if __name__ == '__main__':
    Classifier('models.resnet18', pretrained=True)
    # Classifier('models.resnet50', pretrained=True)