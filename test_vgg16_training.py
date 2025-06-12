import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.dataloader import get_dataloaders, load_config
from models.complex_vgg import complex_vgg16
from models.components.losses import ComplexMagnitudeLoss


def test_vgg16_training():
    print("=== Testing Complex VGG-16 Training ===")

    # 1. Load configuration and a small subset of data
    config = load_config()

    # Make sure config has correct input shape for complex model
    if "data" not in config:
        config["data"] = {}
    if "input_shape" not in config["data"]:
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
    else:
        config["data"]["input_shape"][0] = 3  # Set to 3 channels for complex input

    # Add VGG-specific configuration
    if "model" not in config:
        config["model"] = {}
    config["model"]["dropout_rate"] = 0.5
    config["model"]["activation"] = "modrelu"  # options: modrelu, zrelu, cardioid

    # Make sure to use the complex dataset
    dataloaders = get_dataloaders(dataset_type="complex")

    # Get a small subset of training data
    test_iterations = 3  # Smaller number for VGG since it's larger
    train_subset = []
    for i, (inputs, targets) in enumerate(dataloaders["train"]):
        train_subset.append((inputs, targets))
        if i >= test_iterations - 1:
            break

    # 2. Create complex VGG-16 model and training components
    print("Creating VGG-16 model...")
    model = complex_vgg16(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = ComplexMagnitudeLoss()  # Use complex-compatible loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for VGG

    print(f"Complex VGG-16 model created successfully on {device}")
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    # 3. Run a few training iterations
    model.train()
    training_successful = True
    epoch_loss = 0

    try:
        print("\nStarting training loop...")
        for epoch in range(1):  # Just one epoch for testing
            for i, (inputs, targets) in enumerate(train_subset):
                print(f"Processing batch {i+1}/{len(train_subset)}")
                print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")

                # Move to device
                inputs, targets = inputs.to(device), targets.to(device).float()

                # Forward pass
                optimizer.zero_grad()
                print("Running forward pass...")
                outputs = model(inputs)
                print(f"Output shape: {outputs.shape}, dtype: {outputs.dtype}")

                loss = criterion(outputs.squeeze(), targets)
                print(f"Loss: {loss.item():.6f}")

                # Backward pass
                print("Running backward pass...")
                loss.backward()
                optimizer.step()

                # Track loss
                current_loss = loss.item()
                epoch_loss += current_loss

                print(
                    f"Batch {i+1}/{len(train_subset)} completed. Loss: {current_loss:.6f}"
                )

        print(f"\nTraining complete. Average loss: {epoch_loss/len(train_subset):.6f}")
        print("Complex VGG-16 model training test completed successfully!")
        training_successful = True

    except Exception as e:
        training_successful = False
        print(f"\nError during VGG-16 model training: {e}")
        print("Training test failed. Here's the detailed error:")
        import traceback

        traceback.print_exc()

        print("\nPossible issues to check:")
        print(
            "1. Memory issues - VGG-16 is memory intensive, especially with complex values"
        )
        print("2. Input shape problems - verify tensor dimensions match expected input")
        print(
            "3. Batch normalization implementation - complex batch norm can be unstable"
        )
        print("4. Linear layer implementation - check complex linear operations")

    return training_successful


if __name__ == "__main__":
    test_vgg16_training()
