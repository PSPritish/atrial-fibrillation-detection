import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.architectures.complex_layers import (
    ComplexConv2d,
    ComplexNaiveBatchNorm2d,
    ComplexAdaptiveAvgPool2d,
    ComplexDropout,
)
from models.architectures.complex_activations import ModReLU

# Constants for EfficientNet-B0 (baseline model)
DEFAULT_BLOCKS_ARGS = [
    # repeat, kernel_size, stride, expand_ratio, input_filters, output_filters, se_ratio
    [1, 3, 1, 1, 32, 16, 0.25],
    [2, 3, 2, 6, 16, 24, 0.25],
    [2, 5, 2, 6, 24, 40, 0.25],
    [3, 3, 2, 6, 40, 80, 0.25],
    [3, 5, 1, 6, 80, 112, 0.25],
    [4, 5, 2, 6, 112, 192, 0.25],
    [1, 3, 1, 6, 192, 320, 0.25],
]


class ComplexSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ComplexSEBlock(nn.Module):
    """Complex Squeeze-and-Excitation block"""

    def __init__(self, channels, reduction_ratio=4):
        super(ComplexSEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = ComplexAdaptiveAvgPool2d(1)
        self.fc1 = ComplexConv2d(channels, reduced_channels, kernel_size=1)
        self.activation = ComplexSwish()
        self.fc2 = ComplexConv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = torch.sigmoid(torch.abs(scale))  # Use magnitude for sigmoid
        # Convert to complex scale factor
        scale_complex = scale.type_as(x)
        return x * scale_complex


class ComplexMBConvBlock(nn.Module):
    """Complex Mobile Inverted Bottleneck ConvBlock"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio
    ):
        super(ComplexMBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                ComplexConv2d(
                    in_channels, expanded_channels, kernel_size=1, bias=False
                ),
                ComplexNaiveBatchNorm2d(expanded_channels),
                ModReLU(expanded_channels),
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            ComplexConv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            ComplexNaiveBatchNorm2d(expanded_channels),
            ModReLU(expanded_channels),
        )

        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = ComplexSEBlock(
                expanded_channels, reduction_ratio=expanded_channels // se_channels
            )
        else:
            self.se = nn.Identity()

        # Output phase
        self.project = nn.Sequential(
            ComplexConv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            ComplexNaiveBatchNorm2d(out_channels),
        )

        # Dropout for stochastic depth
        self.dropout = ComplexDropout(p=0.2)

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand(x)

        # Depthwise convolution
        x = self.depthwise(x)

        # Squeeze and Excitation
        x = self.se(x)

        # Output phase
        x = self.project(x)

        # Skip connection
        if self.use_residual:
            x = self.dropout(x)
            x = x + identity

        return x


class ComplexEfficientNet(nn.Module):
    def __init__(
        self,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        dropout_rate=0.2,
        num_classes=1,
    ):
        super(ComplexEfficientNet, self).__init__()

        # Stem
        in_channels = 3  # Complex input has 3 complex channels
        out_channels = self._round_filters(32, width_multiplier)

        self.stem = nn.Sequential(
            ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            ComplexNaiveBatchNorm2d(out_channels),
            ModReLU(out_channels),
        )

        # Build blocks
        self.blocks = nn.Sequential()

        # Initial block input channels
        in_channels = out_channels

        # For each block config
        for i, block_args in enumerate(DEFAULT_BLOCKS_ARGS):
            repeats, kernel_size, stride, expand_ratio, _, filters, se_ratio = (
                block_args
            )

            # Adjust repeats based on depth multiplier
            repeats = int(math.ceil(repeats * depth_multiplier))

            # Adjust filters based on width multiplier
            out_channels = self._round_filters(filters, width_multiplier)

            # First block with stride
            self.blocks.add_module(
                f"block_{i}_0",
                ComplexMBConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    expand_ratio,
                    se_ratio,
                ),
            )

            # Remaining blocks with stride 1
            for j in range(1, repeats):
                self.blocks.add_module(
                    f"block_{i}_{j}",
                    ComplexMBConvBlock(
                        out_channels,
                        out_channels,
                        kernel_size,
                        1,
                        expand_ratio,
                        se_ratio,
                    ),
                )

            in_channels = out_channels

        # Head
        head_channels = self._round_filters(1280, width_multiplier)
        self.head = nn.Sequential(
            ComplexConv2d(in_channels, head_channels, kernel_size=1, bias=False),
            ComplexNaiveBatchNorm2d(head_channels),
            ModReLU(head_channels),
        )

        # Final pooling and dropout
        self.pool = ComplexAdaptiveAvgPool2d(1)
        # Use regular dropout instead of complex dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(head_channels, num_classes))

    def _round_filters(self, filters, width_multiplier):
        """Round number of filters based on width multiplier"""
        return int(filters * width_multiplier)

    def forward(self, x):
        # Handle input conversion to complex if needed
        if not torch.is_complex(x):
            if x.shape[1] == 3:  # Regular RGB image
                print("Converting RGB image to complex...")
                real_part = x[:, 0:1, :, :]
                imag_part = x[:, 1:2, :, :]
                x = torch.complex(real_part.squeeze(1), imag_part.squeeze(1))
            elif x.shape[1] == 6:  # 3 real + 3 imaginary channels
                real_part = x[:, 0:3, :, :]
                imag_part = x[:, 3:6, :, :]
                x = torch.complex(real_part, imag_part)

        # Forward pass
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)

        # Convert to magnitude for classification
        x = torch.abs(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.classifier(x)

        return x


def complex_efficientnet_b0(config=None, num_classes=1):
    """Creates EfficientNet-B0 with complex layers"""
    return ComplexEfficientNet(
        width_multiplier=1.0,
        depth_multiplier=1.0,
        dropout_rate=0.2,
        num_classes=num_classes,
    )


def complex_efficientnet_b1(config=None, num_classes=1):
    """Creates EfficientNet-B1 with complex layers"""
    return ComplexEfficientNet(
        width_multiplier=1.0,
        depth_multiplier=1.1,
        dropout_rate=0.2,
        num_classes=num_classes,
    )


def complex_efficientnet_b2(config=None, num_classes=1):
    """Creates EfficientNet-B2 with complex layers"""
    return ComplexEfficientNet(
        width_multiplier=1.1,
        depth_multiplier=1.2,
        dropout_rate=0.3,
        num_classes=num_classes,
    )
