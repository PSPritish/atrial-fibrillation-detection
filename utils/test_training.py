import torch
import torch.nn as nn
import torch.optim as optim
from utils.tqdmtest import tqdm
from data.dataloader import get_dataloaders, load_config
from models.resnet import AFResNet


def test_training():
    print("=== Testing Model Training ===")

    # 1. Load configuration and a small subset of data
    config = load_config()
    dataloaders = get_dataloaders(dataset_type="combined")

    # Get a small subset of training data
    test_iterations = 5
    train_subset = []
    for i, (inputs, targets) in enumerate(dataloaders["train"]):
        train_subset.append((inputs, targets))
        if i >= test_iterations - 1:
            break

    # 2. Create model and training components
    model = AFResNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model created successfully on {device}")

    # 3. Run a few training iterations
    model.train()
    training_successful = True

    try:
        for epoch in range(2):  # Just 2 epochs
            print(f"\nEpoch {epoch+1}/2:")
            epoch_loss = 0.0

            for i, (inputs, targets) in enumerate(train_subset):
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

        print("\nTraining test completed successfully!")
        print("Your model can train properly.")

    except Exception as e:
        training_successful = False
        print(f"\nError during training: {e}")
        print("Training test failed.")

    return training_successful


if __name__ == "__main__":
    test_training()
