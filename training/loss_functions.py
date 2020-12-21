import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from evaluation.metrics import SMOOTHING_COEFFICIENT


class BinaryDiceLoss(Module):
    """
    Computes the dice loss of each pair of masks in a batch and then average them.
    Masks are expected to be binary (1 and 0).

    dice_loss = (1 - dice_coefficient)
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self,
                preds: Tensor,
                labels: Tensor) -> Tensor:
        """
        :param preds: predicted binary masks
        :param labels: ground truth binary masks
        :return: the mean of (1 - dice_coefficient) for each pair of masks in the batch
        """

        if not torch.is_tensor(preds):
            raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
        if not torch.is_tensor(labels):
            raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')

        if preds.ndim == 4:
            preds = preds.squeeze(1)

        return torch.mean(1. - (2. * (preds * labels).sum((1, 2)) + SMOOTHING_COEFFICIENT) /
                                (labels.sum((1, 2)) + preds.sum((1, 2)) + SMOOTHING_COEFFICIENT))


class BinaryFocalLoss(Module):
    """
    Computes the focal loss of each pair of masks in the batch and then average them.
    Masks are expected to be binary (1 and 0).

    focal_loss = alpha * (1 - pt)^(gamma) * log(pt)
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 gamma: float = 0.0,
                 alpha: float = 1.0):
        """
        :param gamma: the focusing parameter (e.g. 0, 0.5, 1, 2, 5)
        :param alpha: the weight to apply to the foreground class
        """

        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,
                preds: Tensor,
                labels: Tensor) -> Tensor:
        """
        :param preds: predicted binary masks
        :param labels: ground truth binary masks
        :return: the mean of the focal loss of each pair of masks in the batch
        """

        if not torch.is_tensor(preds):
            raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
        if not torch.is_tensor(labels):
            raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')

        if preds.ndim == 4:
            preds = preds.squeeze(1)

        logpt = F.binary_cross_entropy(preds,
                                       labels,
                                       reduction='none')

        pt = torch.clamp(-logpt.exp(), min=-100)
        return torch.mean(self.alpha * ((1 - pt) ** self.gamma) * logpt)
