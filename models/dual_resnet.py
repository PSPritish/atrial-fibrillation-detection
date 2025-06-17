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
        real_output = self.real_resnet(x.real)
        imaginary_output = self.imaginary_resnet(x.imag)
        complex_output = torch.complex(real_output, imaginary_output)
        return complex_output
