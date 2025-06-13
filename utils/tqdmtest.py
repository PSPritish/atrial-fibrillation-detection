import time
import random
import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
from tqdm.notebook import tqdm as notebook_tqdm


def simulate_processing(items, delay=0.01):
    """Simulate processing items with small delay"""
    results = []
    for item in items:
        time.sleep(delay * random.uniform(0.5, 1.5))
        results.append(item * 2)
    return results


def basic_tqdm_demo():
    """Demonstrate basic tqdm usage"""
    print("\n=== Basic TQDM ===")
    items = list(range(100))

    # Standard tqdm
    print("Standard tqdm:")
    for _ in tqdm(items):
        time.sleep(0.01)

    # With description
    print("\nWith description:")
    for _ in tqdm(items, desc="Processing"):
        time.sleep(0.01)

    # With unit
    print("\nWith custom unit:")
    for _ in tqdm(items, desc="Processing", unit="sample"):
        time.sleep(0.01)


def color_tqdm_demo():
    """Demonstrate colored tqdm bars"""
    print("\n=== Colored TQDM Bars ===")
    items = list(range(50))

    # Green bar
    print("Green progress bar:")
    for _ in tqdm(items, desc="Training", colour="green"):
        time.sleep(0.02)

    # Red bar
    print("\nRed progress bar:")
    for _ in tqdm(items, desc="Validation", colour="red"):
        time.sleep(0.02)

    # Blue bar
    print("\nBlue progress bar:")
    for _ in tqdm(items, desc="Testing", colour="blue"):
        time.sleep(0.02)


def format_tqdm_demo():
    """Demonstrate format customization"""
    print("\n=== Custom Formatting ===")
    items = list(range(100))

    # Custom bar format
    print("Custom bar format:")
    bar_format = (
        "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    for _ in tqdm(items, desc="Custom Bar", bar_format=bar_format):
        time.sleep(0.01)

    # With percentage and custom width
    print("\nCustom width (50 chars):")
    for _ in tqdm(items, desc="Fixed Width", ncols=50):
        time.sleep(0.01)

    # With position for multiple bars
    print("\nWith position (for multiple bars):")
    for i in range(3):
        for _ in tqdm(range(30), desc=f"Position {i}", position=i):
            time.sleep(0.03)

    # Reset position
    for i in range(3):
        print()


def nested_tqdm_demo():
    """Demonstrate nested progress bars"""
    print("\n=== Nested Progress Bars ===")

    # Simple nested bars
    print("Nested bars (epochs and batches):")
    for epoch in tqdm(range(5), desc="Epochs"):
        for batch in tqdm(range(20), desc=f"Epoch {epoch+1}", leave=False):
            time.sleep(0.02)

    # Machine learning style
    print("\nML training style:")
    for epoch in tqdm(range(3), desc="Training"):
        # Training phase
        for batch in tqdm(
            range(30), desc=f"Epoch {epoch+1} [Train]", leave=False, colour="green"
        ):
            time.sleep(0.01)

        # Validation phase
        for batch in tqdm(
            range(10), desc=f"Epoch {epoch+1} [Valid]", leave=False, colour="blue"
        ):
            time.sleep(0.03)


def postfix_tqdm_demo():
    """Demonstrate dynamic postfix updates"""
    print("\n=== Dynamic Metrics Update ===")

    # Training loop with metrics
    epochs = 5
    batches = 20

    print("ML training with metrics:")
    for epoch in range(epochs):
        running_loss = 0.0
        accuracy = 0.0

        # Create progress bar
        pbar = tqdm(range(batches), desc=f"Epoch {epoch+1}")

        for batch in pbar:
            # Simulate batch processing
            time.sleep(0.05)

            # Update metrics
            running_loss += random.uniform(-0.1, 0.1)
            loss = max(0, 2.0 - epoch * 0.3 - batch * 0.01 + running_loss)
            accuracy = min(
                1.0,
                (epoch * 10 + batch) / (epochs * batches) * 0.9
                + random.uniform(0, 0.1),
            )

            # Update postfix with metrics
            pbar.set_postfix({"loss": f"{loss:.4f}", "accuracy": f"{accuracy:.4f}"})


def gui_tqdm_demo():
    """Demonstrate GUI mode"""
    print("\n=== GUI Mode ===")
    items = list(range(100))

    # Import the GUI version of tqdm
    from tqdm.gui import tqdm as gui_tqdm

    print("Standard vs GUI mode:")
    print("1. Standard mode:")
    for _ in tqdm(items, desc="Standard"):
        time.sleep(0.01)

    print("\n2. GUI mode:")
    # Use gui_tqdm instead of the gui=True parameter
    for _ in gui_tqdm(items, desc="GUI Mode"):
        time.sleep(0.01)


def auto_tqdm_demo():
    """Demonstrate tqdm.auto which adapts to environment"""
    print("\n=== Auto TQDM (adapts to environment) ===")
    items = list(range(100))

    for _ in auto_tqdm(items, desc="Auto TQDM"):
        time.sleep(0.01)


def compare_styles():
    """Compare different styles side by side"""
    print("\n=== Style Comparison ===")
    n = 50

    # Basic
    for _ in tqdm(range(n), desc="Basic"):
        time.sleep(0.02)

    # With metrics
    pbar = tqdm(range(n), desc="With Metrics")
    for i in pbar:
        time.sleep(0.02)
        pbar.set_postfix({"loss": f"{2.0-i/n:.2f}", "acc": f"{i/n:.2f}"})

    # Compact
    for _ in tqdm(range(n), desc="Compact", ncols=80):
        time.sleep(0.02)

    # Colored
    for _ in tqdm(range(n), desc="Colored", colour="cyan"):
        time.sleep(0.02)

    # Custom format
    custom_fmt = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    for _ in tqdm(range(n), desc="Custom Format", bar_format=custom_fmt):
        time.sleep(0.02)


if __name__ == "__main__":
    print("TQDM Style Experiments")
    print("======================")

    # Uncomment demos you want to run
    basic_tqdm_demo()
    color_tqdm_demo()
    format_tqdm_demo()
    nested_tqdm_demo()
    postfix_tqdm_demo()
    gui_tqdm_demo()
    # auto_tqdm_demo()  # Uncomment if in notebook environment
    compare_styles()

    print("\nExperiments complete!")
