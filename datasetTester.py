import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
from data.dataset import ComplexDataset
from data.transforms import get_transforms
import argparse
from torchvision.io import read_image


def verify_image_loading(mode="train", num_samples=3):
    """Verify image loading from ComplexDataset"""

    # Get transforms
    transforms = get_transforms()

    # Initialize dataset
    dataset = ComplexDataset(mode=mode, transforms=transforms)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Create figure for visualization - expanded to show normalized images
    fig, axes = plt.subplots(num_samples, 9, figsize=(24, 3 * num_samples))

    for i in range(num_samples):
        # Get random index
        idx = np.random.randint(len(dataset))

        # Get original images
        gadf_path, gasf_path, label = dataset.data[idx]
        # Print the paths to verify
        print(f"GADF Path: {gadf_path}")
        print(f"GASF Path: {gasf_path}")
        gadf_image_orig = read_image(gadf_path)  # returns tensor in [C,H,W] format
        gasf_image_orig = read_image(gasf_path)

        # Apply transforms without normalization
        if transforms:
            gadf_tensor = transforms(gadf_image_orig)
            gasf_tensor = transforms(gasf_image_orig)

        # Apply the same normalization as in the dataset
        gadf_tensor_norm = T.functional.normalize(
            gadf_tensor, mean=dataset.gadf_mean, std=dataset.gadf_std
        )
        gasf_tensor_norm = T.functional.normalize(
            gasf_tensor, mean=dataset.gasf_mean, std=dataset.gasf_std
        )

        # Get the complex tensor directly from dataset
        complex_tensor, label = dataset[idx]

        # Calculate magnitude and phase
        magnitude = torch.abs(complex_tensor)
        phase = torch.angle(complex_tensor)

        # Get real and imaginary parts separately
        real_part = complex_tensor.real
        imag_part = complex_tensor.imag

        # Convert tensors to numpy for plotting
        gadf_np = gadf_tensor.permute(1, 2, 0).numpy()
        gasf_np = gasf_tensor.permute(1, 2, 0).numpy()
        gadf_norm_np = gadf_tensor_norm.permute(1, 2, 0).numpy()
        gasf_norm_np = gasf_tensor_norm.permute(1, 2, 0).numpy()
        magnitude_np = magnitude.permute(1, 2, 0).numpy()
        phase_np = phase.permute(1, 2, 0).numpy()
        real_np = real_part.permute(1, 2, 0).numpy()
        imag_np = imag_part.permute(1, 2, 0).numpy()

        # Plot images
        axes[i, 0].imshow(
            gadf_image_orig.permute(1, 2, 0).numpy()
        )  # Convert [C,H,W] to [H,W,C]
        axes[i, 0].set_title(f"Original GADF (Label: {label})")

        axes[i, 1].imshow(
            gasf_image_orig.permute(1, 2, 0).numpy()
        )  # Convert [C,H,W] to [H,W,C]
        axes[i, 1].set_title("Original GASF")

        axes[i, 2].imshow(np.clip(gadf_np, 0, 1))
        axes[i, 2].set_title("Processed GADF")

        axes[i, 3].imshow(np.clip(gasf_np, 0, 1))
        axes[i, 3].set_title("Processed GASF")

        # Add normalized images with coolwarm colormap for comparison
        axes[i, 4].imshow(gadf_norm_np[:, :, 0], cmap="coolwarm")
        axes[i, 4].set_title("Normalized GADF")

        axes[i, 5].imshow(gasf_norm_np[:, :, 0], cmap="coolwarm")
        axes[i, 5].set_title("Normalized GASF")

        # Show real part (R channel)
        axes[i, 6].imshow(real_np[:, :, 0], cmap="coolwarm")
        axes[i, 6].set_title("Real Part (R channel)")

        # Show imaginary part (R channel)
        axes[i, 7].imshow(imag_np[:, :, 0], cmap="coolwarm")
        axes[i, 7].set_title("Imaginary Part (R channel)")

        # Show magnitude (R channel)
        axes[i, 8].imshow(magnitude_np[:, :, 0], cmap="viridis")
        axes[i, 8].set_title("Magnitude (R channel)")

        # Remove axis ticks for clarity
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("complex_visualization.png")
    plt.show()

    print(f"Verification complete. Image saved as 'complex_visualization.png'")

    # Print detailed tensor statistics
    print("\nDataset Statistics:")
    print(f"GASF Mean: {dataset.gasf_mean}")
    print(f"GASF Std: {dataset.gasf_std}")
    print(f"GADF Mean: {dataset.gadf_mean}")
    print(f"GADF Std: {dataset.gadf_std}")

    # Check tensor shapes and types
    print("\nTensor Information:")
    print(f"Complex Tensor Shape: {complex_tensor.shape}")
    print(f"Complex Tensor Type: {complex_tensor.dtype}")
    print(f"Is Complex: {torch.is_complex(complex_tensor)}")

    # Print statistics about real and imaginary parts
    print("\nComplex Component Statistics:")
    print(
        f"Real Part Min/Max: {complex_tensor.real.min().item():.4f}/{complex_tensor.real.max().item():.4f}"
    )
    print(
        f"Imaginary Part Min/Max: {complex_tensor.imag.min().item():.4f}/{complex_tensor.imag.max().item():.4f}"
    )
    print(
        f"Magnitude Min/Max: {magnitude.min().item():.4f}/{magnitude.max().item():.4f}"
    )

    # Also print normalized tensor min/max for comparison
    print(
        f"Normalized GASF Min/Max: {gasf_tensor_norm.min().item():.4f}/{gasf_tensor_norm.max().item():.4f}"
    )
    print(
        f"Normalized GADF Min/Max: {gadf_tensor_norm.min().item():.4f}/{gadf_tensor_norm.max().item():.4f}"
    )

    # Check for NaN or Inf values
    print(
        f"Contains NaN: {torch.isnan(complex_tensor.real).any() or torch.isnan(complex_tensor.imag).any()}"
    )
    print(
        f"Contains Inf: {torch.isinf(complex_tensor.real).any() or torch.isinf(complex_tensor.imag).any()}"
    )

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify image loading from ComplexDataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset mode (train, test, or val)",
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="Number of samples to visualize"
    )

    args = parser.parse_args()
    verify_image_loading(mode=args.mode, num_samples=args.samples)
