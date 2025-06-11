import torch


class ComplexConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2D, self).__init__()
        self.real_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.imag_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, x):
        real = self.real_conv(x[:, 0:1, :, :])  # Real part
        imag = self.imag_conv(x[:, 1:2, :, :])  # Imaginary part
        return torch.cat((real, imag), dim=1)


class ComplexBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = torch.nn.BatchNorm2d(num_features)
        self.imag_bn = torch.nn.BatchNorm2d(num_features)

    def forward(self, x):
        real = self.real_bn(x[:, 0:1, :, :])  # Real part
        imag = self.imag_bn(x[:, 1:2, :, :])  # Imaginary part
        return torch.cat((real, imag), dim=1)


class ComplexActivation(torch.nn.Module):
    def __init__(self, activation_fn=torch.nn.ReLU):
        super(ComplexActivation, self).__init__()
        self.activation_fn = activation_fn()

    def forward(self, x):
        real = self.activation_fn(x[:, 0:1, :, :])  # Real part
        imag = self.activation_fn(x[:, 1:2, :, :])  # Imaginary part
        return torch.cat((real, imag), dim=1)
