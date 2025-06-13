import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data.dataloader import get_dataloaders, load_config
from models.resnet import AFResNet
from models.complex_resnet import complex_resnet18, complex_resnet34, complex_resnet50


def test_standard_resnet(config, test_loader):
    """Test standard ResNet model"""
    print("\n======= TESTING STANDARD RESNET =======")

    # Create model
    model = AFResNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get a batch of data
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        break  # Just use first batch

    # Print input details
    print(f"Input shape: {inputs.shape}")
    print(f"Input type: {inputs.dtype}")
    print(f"Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    # Print output details
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs)
    print(f"Probability range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    # Print a few predictions
    for i in range(min(5, len(targets))):
        print(f"Sample {i}: Target={targets[i].item()}, Pred={probs[i].item():.4f}")

    return model, outputs, probs


def test_complex_resnet(config, test_loader):
    """Test complex ResNet model"""
    print("\n======= TESTING COMPLEX RESNET =======")

    # Create model
    model = complex_resnet18(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Get a batch of data
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        break  # Just use first batch

    # Complex models expect data differently - first 2 channels are real and imaginary parts
    if inputs.shape[1] >= 2:
        print("Using first two channels as real and imaginary parts")
    else:
        print("WARNING: Input doesn't have enough channels for complex data")
        # Duplicate the first channel as imaginary part
        if inputs.shape[1] == 1:
            inputs = torch.cat([inputs, torch.zeros_like(inputs)], dim=1)

    # Print input details
    print(f"Input shape: {inputs.shape}")
    print(f"Input type: {inputs.dtype}")

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    # Print output details
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(outputs)
    print(f"Probability range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    # Print a few predictions
    for i in range(min(5, len(targets))):
        print(f"Sample {i}: Target={targets[i].item()}, Pred={probs[i].item():.4f}")

    return model, outputs, probs


def visualize_features(model, inputs, layer_name=None):
    """Visualize intermediate features"""
    print("\n======= VISUALIZING FEATURES =======")

    # Use hooks to get intermediate activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # Register hooks
    if isinstance(model, AFResNet):
        # For AFResNet
        if layer_name is None:
            layer_name = "layer1"
        model.backbone._modules[layer_name].register_forward_hook(
            get_activation(layer_name)
        )
    else:
        # For ComplexResNet
        if layer_name is None:
            layer_name = "layer1"
        model._modules[layer_name].register_forward_hook(get_activation(layer_name))

    # Forward pass
    with torch.no_grad():
        _ = model(inputs)

    # Get activations
    feature_maps = activations[layer_name]

    # Convert to numpy for visualization
    if torch.is_complex(feature_maps):
        # For complex feature maps, visualize magnitude
        feature_maps = torch.abs(feature_maps)

    # Take first image and first few channels
    feature_maps = feature_maps[0].cpu().numpy()
    print(f"Feature maps shape: {feature_maps.shape}")

    # Plot first 16 feature maps
    num_features = min(16, feature_maps.shape[0])
    plt.figure(figsize=(20, 10))
    for i in range(num_features):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[i], cmap="viridis")
        plt.title(f"Feature {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"feature_maps_{layer_name}.png")
    print(f"Feature maps saved to feature_maps_{layer_name}.png")


def main():
    print("Loading configuration and data...")

    # Load configuration
    config = load_config()

    # Create a copy of original config (for standard model)
    standard_config = config.copy() if isinstance(config, dict) else load_config()

    # Get test dataloader
    dataloaders = get_dataloaders(dataset_type="combined")
    test_loader = dataloaders["test"]

    # Test standard ResNet with original config
    standard_model, standard_outputs, standard_probs = test_standard_resnet(
        standard_config, test_loader
    )

    # Only modify config for complex model
    if config.get("data", {}).get("input_shape", [3, 128, 128])[0] == 3:
        # Assuming first 2 channels will be used for real/imaginary components
        config["data"]["input_shape"][0] = 2

    try:
        # Test complex ResNet with modified config
        complex_model, complex_outputs, complex_probs = test_complex_resnet(
            config, test_loader
        )

        # Visualize features
        print("\nVisualizing standard ResNet features...")
        visualize_features(standard_model, next(iter(test_loader))[0], "layer2")

        print("\nVisualizing complex ResNet features...")
        visualize_features(complex_model, next(iter(test_loader))[0], "layer2")

    except Exception as e:
        print(f"Error testing complex model: {e}")
        print(
            "Complex model test skipped. Make sure all components are properly implemented."
        )


if __name__ == "__main__":
    main()
