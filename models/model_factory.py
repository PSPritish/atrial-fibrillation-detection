from models.architectures.resnet import ResNet
from models.complex_resnet import ComplexResNet


def create_model(config):
    model_type = config["model"]["type"]
    if model_type == "resnet18":
        return ResNet(depth=18, num_classes=config["model"]["num_classes"])
    elif model_type == "resnet34":
        return ResNet(depth=34, num_classes=config["model"]["num_classes"])
    elif model_type == "resnet50":
        return ResNet(depth=50, num_classes=config["model"]["num_classes"])
    elif model_type == "complex_resnet50":
        return ComplexResNet(depth=50, num_classes=config["model"]["num_classes"])
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
