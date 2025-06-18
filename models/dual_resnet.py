import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DualResNet(nn.Module):
    def __init__(self, num_classes):
        super(DualResNet, self).__init__()
        if num_classes is None:
            raise ValueError("num_classes must be specified")

        self.real_resnet = models.resnet18(weights=None)
        self.imaginary_resnet = models.resnet18(weights=None)
        self.real_resnet.fc = nn.Linear(self.real_resnet.fc.in_features, num_classes)
        self.imaginary_resnet.fc = nn.Linear(
            self.imaginary_resnet.fc.in_features, num_classes
        )

    def forward(self, x):
        # Handle both complex tensor and tensor with extra dimension for real/imag parts
        if x.dim() == 5:  # When using DataParallel, complex tensor becomes [B, C, H, W, 2]
            real_part = x[..., 0]  # Extract real component
            imag_part = x[..., 1]  # Extract imaginary component
        else:  # Standard complex tensor format
            real_part = x.real
            imag_part = x.imag

        real_output = self.real_resnet(real_part)
        imaginary_output = self.imaginary_resnet(imag_part)

        # Instead of returning a complex tensor, return a concatenated tensor
        # where the last dimension contains real and imaginary parts
        # This is compatible with DataParallel
        return torch.stack([real_output, imaginary_output], dim=-1)
