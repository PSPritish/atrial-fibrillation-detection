import torch
import torch.nn as nn
import torchvision.models as models


class AFResNet(nn.Module):
    def __init__(self, config):
        super(AFResNet, self).__init__()
        # Get model parameters from config
        architecture = config.get("model", {}).get("architecture", "ResNet-18")

        # Get input channels from config
        input_channels = config.get("data", {}).get("input_shape", [3, 128, 128])[0]

        # Load pre-trained model with 3 channels
        if architecture == "ResNet-18":
            self.backbone = models.resnet18(weights=None)
        elif architecture == "ResNet-34":
            self.backbone = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Modify the first convolutional layer if needed
        if input_channels != 3:
            # Replace first layer to accept different number of channels
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace final layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)
