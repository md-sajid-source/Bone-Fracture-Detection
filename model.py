

import torch.nn as nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()

        if backbone == 'resnet18':
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )

        elif backbone == 'resnet34':
            self.model = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT
            )

        elif backbone == 'resnet50':
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        # Binary classification:
        # 0 = Not Fractured
        # 1 = Fractured
        self.model.fc = nn.Linear(
            self.model.fc.in_features, 2
        )

    def forward(self, x):
        return self.model(x)
