import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class ComplexAwareBCE(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexAwareBCE, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assuming inputs are complex-valued and targets are binary
        real_inputs = inputs.real
        imag_inputs = inputs.imag
        real_targets = targets.real
        imag_targets = targets.imag

        bce_real = nn.BCEWithLogitsLoss(reduction=self.reduction)(real_inputs, real_targets)
        bce_imag = nn.BCEWithLogitsLoss(reduction=self.reduction)(imag_inputs, imag_targets)

        if self.reduction == 'mean':
            return (bce_real + bce_imag) / 2
        elif self.reduction == 'sum':
            return bce_real + bce_imag
        else:
            return bce_real, bce_imag