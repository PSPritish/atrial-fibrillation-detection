import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DualResNetCF(nn.Module):
    def __init__(self, num_classes):
        super(DualResNetCF, self).__init__()

        self.real_resnet = models.resnet18()
        self.imaginary_resnet = models.resnet18()
        # Discard the fully connected layers
        self.real_resnet.fc = nn.Identity()
        self.imaginary_resnet.fc = nn.Identity()

        # Get feature dimension (512 for ResNet18)
        feature_dim = 512

        # Create new classifier for the combined features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        real_features = self.real_resnet(x.real)
        imaginary_features = self.imaginary_resnet(x.imag)

        # Concatenate features from both networks
        combined_features = torch.cat((real_features, imaginary_features), dim=1)

        # Pass through the classifier
        return self.classifier(combined_features)
