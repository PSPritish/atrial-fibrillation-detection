import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class ComplexMagnitudeLoss(nn.Module):
    """Loss function for complex models that output magnitude before final layer"""

    def __init__(self, base_loss="bce", reduction="mean"):
        super(ComplexMagnitudeLoss, self).__init__()
        self.reduction = reduction
        self.base_loss = base_loss

    def forward(self, inputs, targets):
        # Inputs are already real (magnitude was taken in model)
        # Just apply standard loss
        if self.base_loss == "bce":
            return nn.BCEWithLogitsLoss(reduction=self.reduction)(inputs, targets)
        elif self.base_loss == "focal":
            return FocalLoss(reduction=self.reduction)(inputs, targets)
        else:
            raise ValueError(f"Unsupported base loss: {self.base_loss}")


class ComplexMagnitudeAndPhaseLoss(nn.Module):
    """Loss that considers both magnitude and phase of complex outputs"""

    def __init__(self, phase_weight=0.3, reduction="mean"):
        super(ComplexMagnitudeAndPhaseLoss, self).__init__()
        self.reduction = reduction
        self.phase_weight = phase_weight  # Weight for the phase component

    def forward(self, complex_outputs, targets):
        # This gets complex outputs BEFORE magnitude is taken
        # For use with a modified model that preserves complex values

        # Magnitude component (primary signal strength)
        magnitude = torch.abs(complex_outputs)
        magnitude_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)(
            magnitude, targets
        )

        # Phase component (directional information)
        phase = torch.angle(complex_outputs)
        # For positive samples, phase should be clustered (low variance)
        # For negative samples, phase is less important

        # Target-dependent phase regularization
        positive_samples = (targets == 1).float()
        phase_consistency = (1 - torch.cos(phase)) * positive_samples

        if self.reduction == "mean":
            phase_loss = phase_consistency.mean()
        elif self.reduction == "sum":
            phase_loss = phase_consistency.sum()

        # Combined loss
        return magnitude_loss + self.phase_weight * phase_loss


class ComplexDiceLoss(nn.Module):
    """Dice loss for complex-valued outputs (after magnitude)"""

    def __init__(self, smooth=1.0):
        super(ComplexDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class ComplexFocalLoss(nn.Module):
    """Focal loss specifically designed for complex models"""

    def __init__(self, alpha=1.0, gamma=2.0, magnitude_only=True, reduction="mean"):
        super(ComplexFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.magnitude_only = magnitude_only

    def forward(self, inputs, targets):
        # If we're only using magnitude (default)
        if self.magnitude_only or not torch.is_complex(inputs):
            # This is for when magnitude was already taken in the model
            BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduction == "mean":
                return F_loss.mean()
            elif self.reduction == "sum":
                return F_loss.sum()
            else:
                return F_loss
        else:
            # This is for when we have complex outputs directly
            # Take magnitude first
            magnitude = torch.abs(inputs)
            BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(magnitude, targets)
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduction == "mean":
                return F_loss.mean()
            elif self.reduction == "sum":
                return F_loss.sum()
            else:
                return F_loss


# Example usage in your trainer
def _get_loss_function(self):
    loss_name = self.config.get("training", {}).get("loss_function", "bce")
    is_complex = self.config.get("model", {}).get("complex", False)

    if is_complex:
        if loss_name == "focal_loss":
            return ComplexFocalLoss()
        elif loss_name == "dice_loss":
            return ComplexDiceLoss()
        elif loss_name == "magnitude_phase_loss":
            return ComplexMagnitudeAndPhaseLoss()
        else:
            return ComplexMagnitudeLoss()
    else:
        if loss_name == "focal_loss":
            return FocalLoss()
        elif loss_name == "dice_loss":
            from models.components.losses import DiceLoss

            return DiceLoss()
        else:
            return nn.BCEWithLogitsLoss()
