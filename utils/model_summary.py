import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from tabulate import tabulate  # You may need to install this: pip install tabulate


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size, batch_size=-1, device=None, print_summary=True):
    """
    Generates a summary of the PyTorch model similar to Keras model.summary()

    Args:
        model: PyTorch model instance
        input_size: Input size (tuple or list) excluding batch dimension
        batch_size: Batch size to use for shape inference
        device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        print_summary: Whether to print the summary

    Returns:
        summary: A dictionary containing the summary information
    """

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]["nb_params"] = params

            # For complex layers
            if "Complex" in class_name:
                # Attempt to capture real and imaginary parameters separately
                real_params = 0
                imag_params = 0
                if hasattr(module, "real_conv"):
                    real_params += sum(p.numel() for p in module.real_conv.parameters())
                if hasattr(module, "imag_conv"):
                    imag_params += sum(p.numel() for p in module.imag_conv.parameters())
                if real_params > 0 or imag_params > 0:
                    summary[m_key][
                        "complex_params"
                    ] = f"R:{real_params}, I:{imag_params}"

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # Determine device to use
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            device = "cpu"

    # Move model to the specified device
    model = model.to(device)

    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Batch size of 2 for batchnorm
    x = [torch.rand(2, *in_size).to(device) for in_size in input_size]

    # Check if model is complex
    is_complex_model = any("Complex" in str(type(m)) for m in model.modules())
    if is_complex_model:
        x = [torch.complex(t, torch.zeros_like(t)) for t in x]

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(register_hook)

    # Make a forward pass
    model(*x)

    # Remove these hooks
    for h in hooks:
        h.remove()

    # Create summary table
    table = []
    total_params = 0
    total_output = 0
    trainable_params = 0

    # Header
    header = ["Layer (type)", "Output Shape", "Param #"]

    # Add rows
    for layer in summary:
        table.append(
            [
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            ]
        )
        total_params += summary[layer]["nb_params"]

        if "output_shape" in summary[layer]:
            output_size = summary[layer]["output_shape"]
            if isinstance(output_size, list):
                for item in output_size:
                    total_output += np.prod(item)
            else:
                total_output += np.prod(output_size)

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

    # Add totals
    table.append(["", "", ""])
    table.append(["Total params", "", "{0:,}".format(total_params)])
    table.append(["Trainable params", "", "{0:,}".format(trainable_params)])
    table.append(
        ["Non-trainable params", "", "{0:,}".format(total_params - trainable_params)]
    )

    if print_summary:
        print(tabulate(table, headers=header, tablefmt="grid"))

        # Additional model information
        print(
            f"\nModel Type: {'Complex-valued' if is_complex_model else 'Real-valued'}"
        )
        print(f"Device: {device}")
        memory_usage = total_output * 4  # Bytes
        if is_complex_model:
            memory_usage *= 2  # Complex values use twice the memory
        print(f"Estimated memory usage: {memory_usage / 1024 / 1024:.2f} MB")

    return summary


def test_model_summary(model_name="complex_resnet18"):
    """Test the model summary utility with different models"""
    from models.resnet import AFResNet
    from models.complex_resnet import complex_resnet18, complex_resnet34
    from data.dataloader import load_config

    config = load_config()

    if model_name == "resnet":
        model = AFResNet(config)
        input_size = (3, 128, 128)
    elif model_name == "complex_resnet18":
        # Set up complex model config
        if "data" not in config:
            config["data"] = {}
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
        model = complex_resnet18(config)
        input_size = (3, 128, 128)
    elif model_name == "complex_resnet34":
        if "data" not in config:
            config["data"] = {}
        config["data"]["input_shape"] = [3, 128, 128]  # 3 complex channels
        model = complex_resnet34(config)
        input_size = (3, 128, 128)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"=== Model Summary for {model_name} ===")
    model_summary(model, input_size, device="cpu")

    # Also print total parameter count
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_model_summary(sys.argv[1])
    else:
        test_model_summary()
