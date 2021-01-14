from typing import Dict, Union, List

import numpy as np

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

    def __init__(self, logits: bool = True):
        """
        :param logits: if True it expects predictions as logits, so it passes them into a sigmoid function
        """

        super(BinaryDiceLoss, self).__init__()
        self.logits = logits

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

        # Apply sigmoid if the network outputs are logits
        if self.logits:
            preds = torch.sigmoid(preds)

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
                 logits: bool = True,
                 gamma: float = 0.0,
                 alpha: float = 0.5,
                 device: str = 'cuda'):
        """
        :param logits: if True it expects predictions as logits, so it uses binary_cross_entropy_with_logits
        :param gamma: the focusing parameter (e.g. 0, 0.5, 1, 2, 5)
        :param alpha: the weight to apply to the foreground class
        :param device: PyTorch device
        """

        super(BinaryFocalLoss, self).__init__()
        self.logits = logits
        self.gamma = gamma
        self.alpha = torch.tensor([1 - alpha, alpha]).to(device)

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

        if self.logits:
            logpt = F.binary_cross_entropy_with_logits(preds,
                                                       labels.type_as(preds),
                                                       reduction='none')
        else:
            logpt = F.binary_cross_entropy(preds,
                                           labels.type_as(preds),
                                           reduction='none')

        # Weights depend on the class
        at = self.alpha.gather(0, labels.view(-1))

        pt = torch.clamp(-logpt.exp(), min=-100).view(-1)
        return torch.mean(at * ((1 - pt) ** self.gamma) * logpt.view(-1))


class BinaryLovaszLoss(Module):
    """
    Computes the Lovasz loss for the binary case, called also Lovasz hinge as described in:
    - https://arxiv.org/pdf/1705.08790.pdf
    - http://proceedings.mlr.press/v37/yub15.pdf

    Code adapted from the original repository:
    https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
    """

    def __init__(self):
        super(BinaryLovaszLoss, self).__init__()

    def lovasz_grad(self, gt_sorted: Tensor) -> Tensor:
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors

        :param gt_sorted: labels sorted according to the errors
        :return the Lovasz gradient. This vector says what is the effect of an error on the evolution of the Jaccard
        index. The idea is to minimize the errors that penalize the Jaccard index the most.
        """
        gt_length = len(gt_sorted)

        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        # jaccard contains the evolution of the Jaccard index with respect to the sorted errors
        # It varies between 0 and the actual Jaccard index of the prediction
        jaccard = 1. - intersection / union
        # Exclude the 1-pixel case when the subtraction would be erroneous
        if gt_length > 1:
            jaccard[1:gt_length] = jaccard[1:gt_length] - jaccard[0:-1]
        return jaccard

    def forward(self,
                preds: Tensor,
                labels: Tensor) -> Tensor:
        """
        :param preds: predicted binary masks as logits from the neural network
        :param labels: ground truth binary masks
        :return: the mean of the focal loss of each pair of masks in the batch
        """

        if not torch.is_tensor(preds):
            raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
        if not torch.is_tensor(labels):
            raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')

        # Flat tensors
        preds = preds.view(-1)
        labels = labels.view(-1)

        # Map zeroes labels to -1 and 1 to 1
        # Background has negative sign while foreground has positive sign
        signs = 2. * labels.float() - 1.
        # Errors are positive when the predictions are false and negative when the predictions are true
        # The 1. is a margin used to consider true predictions with a value lower than 1 as errors
        # So at the end errors are positive when predictions are false or correct but with a value lower than 1.,
        # errors are negative otherwise.
        errors = (1. - preds * signs)

        # Sort errors in descending order
        errors_sorted, perm = torch.sort(errors,
                                         dim=0,
                                         descending=True)
        # Sort labels according to the related error
        perm = perm.data
        gt_sorted = labels[perm]
        # Computes the gradient approximation as a backward difference
        grad = self.lovasz_grad(gt_sorted)
        # ReLU is used because only a positive part of the error vector contains prediction errors
        return torch.dot(F.relu(errors_sorted), grad)


class CombinationLoss(Module):
    """
    Weighted averages of multiple losses.
    """

    def __init__(self, loss_functions_weights: List[Dict[str, Union[float, Module]]]):
        """

        :param loss_functions_weights: dictionary containing loss functions and their weights
        For example [{'name': LossFunction1, 'weight': 0.3}, {'name': LossFunction2, 'weight': 0.7}]
        """
        super(CombinationLoss, self).__init__()
        if np.sum(list(map(lambda x: x['weight'], loss_functions_weights))) != 1:
            raise ValueError('Weights sum is not 1!')
        self.loss_functions_weights = loss_functions_weights

    def forward(self,
                preds: Tensor,
                labels: Tensor) -> Tensor:
        """
        :param preds: predicted binary masks as logits from the neural network
        :param labels: ground truth binary masks
        :return: the averaged sum of the given loss functions
        """
        final_loss = None

        for loss_params in self.loss_functions_weights:
            new_loss = loss_params['name'](preds, labels) * loss_params['weight']
            if final_loss is not None:
                final_loss += new_loss
            else:
                final_loss = new_loss

        return final_loss


class HookNetLoss(Module):
    """
    Loss described in the paper which combines the binary cross entropy loss from the context branch the one from the
    target image.
    """

    def __init__(self,
                 logits: bool = True,
                 target_importance=0.75,
                 gamma: float = 0.0,
                 alpha: float = 0.5,
                 device: str = 'cuda'):
        """
        :param logits: if True it expects predictions as logits, so it passes them into a sigmoid function
        :param target_importance: weight applied to the target loss. (1 - target_importance) is applied to the context
        loss.
        :param gamma: the focusing parameter (e.g. 0, 0.5, 1, 2, 5)
        :param alpha: the weight to apply to the foreground class
        :param device: PyTorch device
        """

        super(HookNetLoss, self).__init__()
        assert 0 <= target_importance <= 1, "Target importance must be a weight between 0 and 1."

        self.logits = logits
        self.target_importance = target_importance
        self.ce_loss = BinaryFocalLoss(logits,
                                       gamma=gamma,
                                       alpha=alpha,
                                       device=device)

    def forward(self,
                preds: Union[tuple, list],
                labels: Union[tuple, list]) -> Tensor:
        """
        :param preds: predicted binary masks from target and context branches
        :param labels: ground truth binary masks from target and context datasets
        :return: the mean of (1 - dice_coefficient) for each pair of masks in the batch
        """

        if not type(preds) in [tuple, list]:
            raise TypeError(f'Predictions type is not a tuple of torch.Tensor. Got {format(type(preds))}')
        if not type(labels) in [tuple, list]:
            raise TypeError(f'Labels type is not a tuple of torch.Tensor. Got {format(type(labels))}')

        target_label = labels[0]
        context_label = labels[1]

        # Apply sigmoid if the network outputs are logits
        if self.logits:
            target_pred = torch.sigmoid(preds[0])
            context_pred = torch.sigmoid(preds[1])
        else:
            target_pred = preds[0]
            context_pred = preds[1]

        return self.target_importance * self.ce_loss(target_pred, target_label) + \
               (1 - self.target_importance) * self.ce_loss(context_pred, context_label)
