import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    history (dict): A dictionary containing training history with keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    """
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_complex_metrics(metrics):
    """
    Plots complex-valued specific metrics.

    Parameters:
    metrics (dict): A dictionary containing complex metrics to visualize.
    """
    plt.figure(figsize=(10, 5))

    for key, value in metrics.items():
        plt.plot(value, label=key)

    plt.title('Complex-Valued Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()