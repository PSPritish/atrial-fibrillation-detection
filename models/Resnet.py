import torch
import torch.nn as nn
import torchvision.models as models


class AFResNet(nn.Module):
    def __init__(self, config):
        super(AFResNet, self).__init__()
        # Get model parameters from config
        architecture = config.get("model", {}).get("architecture", "ResNet-18")
        num_classes = 1  # Binary classification

        # Load pre-trained model
        if architecture == "ResNet-18":
            self.backbone = models.resnet18(weights=None)
        elif architecture == "ResNet-34":
            self.backbone = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Modify for GASF/GADF input
        in_channels = config.get("data", {}).get("input_shape", [3, 128, 128])[0]
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
