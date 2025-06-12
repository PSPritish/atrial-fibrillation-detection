import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.dataloader import get_dataloaders, load_config
from models.complex_resnet import complex_resnet18
from models.components.losses import ComplexMagnitudeLoss


def test_complex_training():
    print("=== Testing Complex Model Training ===")

    # 1. Load configuration and a small subset of data
    config = load_config()

    # Make sure config has correct input shape for complex model
    # Set to 3 channels to match our ComplexDataset output
    if "data" not in config:
        config["data"] = {}
    if "input_shape" not in config["data"]:
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
    else:
        config["data"]["input_shape"][0] = 3  # Changed to 3 for 3-channel complex input

    # Make sure to use the complex dataset
    dataloaders = get_dataloaders(dataset_type="complex")

    # Get a small subset of training data
    test_iterations = 5
    train_subset = []
    for i, (inputs, targets) in enumerate(dataloaders["train"]):
        train_subset.append((inputs, targets))
        if i >= test_iterations - 1:
            break

    # 2. Create complex model and training components
    model = complex_resnet18(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = ComplexMagnitudeLoss()  # Use complex-compatible loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Complex model created successfully on {device}")

    # 3. Run a few training iterations
    model.train()
    training_successful = True
    epoch_loss = 0  # Initialize epoch_loss variable

    try:
        for i, (inputs, targets) in enumerate(train_subset):
            # No need to convert to complex - already complex from dataset
            print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")

            # Move to device
            inputs, targets = inputs.to(device), targets.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            current_loss = loss.item()
            epoch_loss += current_loss

            print(f"  Batch {i+1}/{len(train_subset)}, Loss: {current_loss:.6f}")

        print(f"  Epoch Loss: {epoch_loss/len(train_subset):.6f}")
        print("\nComplex model training test completed successfully!")
        print("Your complex model can train properly.")
        training_successful = True

    except Exception as e:
        training_successful = False
        print(f"\nError during complex model training: {e}")
        print("Training test failed. Here's the detailed error:")
        import traceback

        traceback.print_exc()

        print("\nPossible solutions:")
        print("1. Check that ComplexConv2d properly handles 3-channel complex tensors")
        print("2. Verify that model initialization uses 3 input channels")
        print(
            "3. Ensure apply_complex function works with multiple channel complex inputs"
        )

    return training_successful


if __name__ == "__main__":
    test_complex_training()
