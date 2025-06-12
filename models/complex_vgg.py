import torch
import torch.nn as nn
from models.architectures.complex_layers import (
    ComplexConv2d,
    ComplexMaxPool2d,
    ComplexAdaptiveAvgPool2d,
    ComplexDropout,
    ComplexNaiveBatchNorm2d,
)
from models.architectures.complex_activations import ModReLU, zReLU, ComplexCardioid


class ComplexVGG16(nn.Module):
    def __init__(self, config, num_classes=1, init_weights=True):
        super(ComplexVGG16, self).__init__()
        # Get configuration parameters
        input_channels = config.get("data", {}).get("input_shape", [3, 128, 128])[0]
        dropout_rate = config.get("model", {}).get("dropout_rate", 0.5)
        activation = config.get("model", {}).get("activation", "modrelu")

        # Select activation function
        if activation == "modrelu":
            self.activation = ModReLU
        elif activation == "zrelu":
            self.activation = zReLU
        elif activation == "cardioid":
            self.activation = ComplexCardioid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # VGG16 feature layers
        self.features = self._make_layers(input_channels)

        # VGG16 classifier layers
        self.avgpool = ComplexAdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            ComplexDropout(p=dropout_rate),
            self._complex_linear(512 * 7 * 7, 4096),
            self.activation(4096),
            ComplexDropout(p=dropout_rate),
            self._complex_linear(4096, 4096),
            self.activation(4096),
            self._complex_linear(4096, num_classes),
        )

        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # Ensure input is complex
        if not torch.is_complex(x):
            # If input is real tensor with channels as [real, imag, ...],
            # convert to complex tensor
            if x.shape[1] >= 2:
                real_part = x[:, 0:1, :, :]
                imag_part = x[:, 1:2, :, :]
                complex_input = torch.complex(
                    real_part.squeeze(1), imag_part.squeeze(1)
                )
                # Add channel dimension back
                x = complex_input.unsqueeze(1)
            else:
                # If only one channel, use as real part and set imaginary to zero
                x = torch.complex(x, torch.zeros_like(x))

        # Forward pass through feature extractor
        x = self.features(x)

        # Forward pass through classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)

        # For binary classification, we take magnitude of complex output
        if self.classifier[-1].out_features == 1:
            # Return real part for compatibility with BCEWithLogitsLoss
            return x.abs()

        return x

    def _make_layers(self, in_channels):
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        layers = []

        for v in cfg:
            if v == "M":
                layers += [ComplexMaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = ComplexConv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, ComplexNaiveBatchNorm2d(v), self.activation(v)]
                in_channels = v

        return nn.Sequential(*layers)

    def _complex_linear(self, in_features, out_features):
        # Create a complex-compatible linear layer using two real linear layers
        class ComplexLinear(nn.Module):
            def __init__(self, in_features, out_features):
                super(ComplexLinear, self).__init__()
                self.real_linear = nn.Linear(in_features, out_features)
                self.imag_linear = nn.Linear(in_features, out_features)

            def forward(self, x):
                if not torch.is_complex(x):
                    raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

                return torch.complex(
                    self.real_linear(x.real) - self.imag_linear(x.imag),
                    self.real_linear(x.imag) + self.imag_linear(x.real),
                )

        return ComplexLinear(in_features, out_features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ComplexConv2d):
                # Initialize real and imaginary convolutions
                nn.init.kaiming_normal_(
                    m.real_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.kaiming_normal_(
                    m.imag_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.real_conv.bias is not None:
                    nn.init.constant_(m.real_conv.bias, 0)
                    nn.init.constant_(m.imag_conv.bias, 0)
            elif hasattr(m, "real_linear"):
                # Initialize complex linear layers
                nn.init.normal_(m.real_linear.weight, 0, 0.01)
                nn.init.normal_(m.imag_linear.weight, 0, 0.01)
                nn.init.constant_(m.real_linear.bias, 0)
                nn.init.constant_(m.imag_linear.bias, 0)
            elif isinstance(m, ComplexNaiveBatchNorm2d):
                # Initialize batch norm layers
                if hasattr(m, "real_bn"):
                    nn.init.constant_(m.real_bn.weight, 1)
                    nn.init.constant_(m.real_bn.bias, 0)
                    nn.init.constant_(m.imag_bn.weight, 0)
                    nn.init.constant_(m.imag_bn.bias, 0)


def complex_vgg16(config):
    """
    Create a complex-valued VGG-16 model

    Args:
        config: Configuration dictionary

    Returns:
        ComplexVGG16: Complex-valued VGG-16 model
    """
    num_classes = 1  # Binary classification for atrial fibrillation
    model = ComplexVGG16(config, num_classes=num_classes)

    print(
        f"Creating complex VGG-16 with {config.get('data', {}).get('input_shape', [3, 128, 128])[0]} complex channels"
    )

    return model


def complex_vgg16_bn(config):
    """
    Alias for complex_vgg16 (all complex VGG models use batch normalization)
    """
    return complex_vgg16(config)
