import torch
import torch.nn as nn


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (
        fr(input.imag) + fi(input.real)
    ).type(dtype)


class ComplexConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(ComplexConv2d, self).__init__()
        self.real_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.imag_conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, x):
        if not torch.is_complex(x):
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")
        return apply_complex(self.real_conv, self.imag_conv, x)


class ComplexMaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

        self.max_pool = torch.nn.MaxPool2d(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode,
        )

    def forward(self, x):

        # check if the input is complex
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor, Got {x.dtype}")

        return (self.max_pool(x.real)).type(torch.complex64) + 1j * (
            self.max_pool(x.imag)
        ).type(torch.complex64)


class ComplexAvgPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super(ComplexAvgPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

        self.avg_pool = torch.nn.AvgPool2d(
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def forward(self, x):
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

        return (self.avg_pool(x.real)).type(torch.complex64) + 1j * (
            self.avg_pool(x.imag)
        ).type(torch.complex64)


class ComplexAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.output_size = output_size

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, input):
        return (self.adaptive_pool(input.real)).type(torch.complex64) + 1j * (
            self.adaptive_pool(input.imag)
        ).type(torch.complex64)


class ComplexDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.real_drop = torch.nn.Dropout(self.p)
        self.imag_drop = torch.nn.Dropout(self.p)

    def forward(self, input):
        return (self.real_drop(input.real)).type(torch.complex64) + 1j * (
            self.imag_drop(input.imag)
        ).type(torch.complex64)


class ComplexNaiveBatchNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device

        self.real_bn = torch.nn.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )
        self.imag_bn = torch.nn.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )

    def forward(self, input):
        # check if the input is a complex tensor
        if not input.is_complex():
            raise ValueError(f"Input should be complex, Got {input.dtype}")

        return (self.real_bn(input.real)).type(torch.complex64) + 1j * (
            self.imag_bn(input.imag)
        ).type(torch.complex64)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters (W matrix and B vector)
        self.Wrr = nn.Parameter(torch.ones(num_features))
        self.Wri = nn.Parameter(torch.zeros(num_features))
        self.Wii = nn.Parameter(torch.ones(num_features))
        self.Br = nn.Parameter(torch.zeros(num_features))
        self.Bi = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer("RMr", torch.zeros(num_features))
        self.register_buffer("RMi", torch.zeros(num_features))
        self.register_buffer("RVrr", torch.ones(num_features))
        self.register_buffer("RVri", torch.zeros(num_features))
        self.register_buffer("RVii", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x: (B, C, H, W) — complex tensor
        xr, xi = x.real, x.imag  # split real and imaginary

        B, C, H, W = xr.shape
        dims = (0, 2, 3)  # reduce over batch and spatial dims

        # Update running stats if training
        if self.training:
            # Mean
            Mr = xr.mean(dim=dims, keepdim=True)
            Mi = xi.mean(dim=dims, keepdim=True)

            # Centered input
            xr_centered = xr - Mr
            xi_centered = xi - Mi

            # Covariance elements
            Vrr = (xr_centered**2).mean(dim=dims, keepdim=True)
            Vri = (xr_centered * xi_centered).mean(dim=dims, keepdim=True)
            Vii = (xi_centered**2).mean(dim=dims, keepdim=True)

            # Update running stats
            self.num_batches_tracked += 1
            m = self.momentum

            self.RMr = (1 - m) * self.RMr + m * Mr.view(C)
            self.RMi = (1 - m) * self.RMi + m * Mi.view(C)
            self.RVrr = (1 - m) * self.RVrr + m * Vrr.view(C)
            self.RVri = (1 - m) * self.RVri + m * Vri.view(C)
            self.RVii = (1 - m) * self.RVii + m * Vii.view(C)
        else:
            # Use running statistics
            Mr = self.RMr.view(1, C, 1, 1)
            Mi = self.RMi.view(1, C, 1, 1)
            xr_centered = xr - Mr
            xi_centered = xi - Mi

            Vrr = self.RVrr.view(1, C, 1, 1)
            Vri = self.RVri.view(1, C, 1, 1)
            Vii = self.RVii.view(1, C, 1, 1)

        # Add small eps for numerical stability
        Vrr = Vrr + self.eps
        Vii = Vii + self.eps

        # Compute inverse square root of 2x2 covariance matrix
        tau = Vrr + Vii
        delta = Vrr * Vii - Vri**2
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()
        rst = 1.0 / (s * t)

        # Whitening matrix U = V^(-0.5)
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = -Vri * rst

        # Apply affine transform: Z = WU, y = Zx + B
        Wrr = self.Wrr.view(1, C, 1, 1)
        Wri = self.Wri.view(1, C, 1, 1)
        Wii = self.Wii.view(1, C, 1, 1)

        Zrr = Wrr * Urr + Wri * Uri
        Zri = Wrr * Uri + Wri * Uii
        Zir = Wri * Urr + Wii * Uri
        Zii = Wri * Uri + Wii * Uii

        yr = Zrr * xr_centered + Zri * xi_centered + self.Br.view(1, C, 1, 1)
        yi = Zir * xr_centered + Zii * xi_centered + self.Bi.view(1, C, 1, 1)

        return torch.complex(yr, yi)
