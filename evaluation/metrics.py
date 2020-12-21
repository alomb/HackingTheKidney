import torch
from torch import Tensor

# Smoothing value to avoid undefined and undetermined operations during the computations
SMOOTHING_COEFFICIENT = 1e-6


def iou(preds: Tensor,
        labels: Tensor) -> Tensor:
    """
    Compute IoU scores given batches of predictions and labels
    The IoU is referred also as the Jaccard index.

    :param preds: batch of binary mask predictions
    :param labels: batch of binary mask labels
    :return: the IoU score for each pair in the batch
    """

    if not torch.is_tensor(preds):
        raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
    if not torch.is_tensor(labels):
        raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')
    if preds.dtype != torch.long:
        raise TypeError(f'Predictions should be tensors of type torch.long. Got {format(preds.dtype)}')
    if labels.dtype != torch.long:
        raise TypeError(f'Labels should be tensors of type torch.long. Got {format(labels.dtype)}')

    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    intersections = (preds & labels).long().sum((1, 2))
    unions = (preds | labels).long().sum((1, 2))

    return (intersections + SMOOTHING_COEFFICIENT) / (unions + SMOOTHING_COEFFICIENT)


def dice_coefficient(preds: Tensor,
                     labels: Tensor) -> Tensor:
    """
    Compute the dice coefficient given batches of predictions and labels.
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient.
    This is the metric used in the competition.

    :param preds: batch of binary mask predictions
    :param labels: batch of binary mask labels
    :return: the dice coefficient for each pair in the batch
    """

    if not torch.is_tensor(preds):
        raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
    if not torch.is_tensor(labels):
        raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')
    if preds.dtype != torch.long:
        raise TypeError(f'Predictions should be tensors of type torch.long. Got {format(preds.dtype)}')
    if labels.dtype != torch.long:
        raise TypeError(f'Labels should be tensors of type torch.long. Got {format(labels.dtype)}')

    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    return (2. * (preds * labels).long().sum((1, 2)) + SMOOTHING_COEFFICIENT) /\
           (labels.sum((1, 2)) + preds.sum((1, 2)) + SMOOTHING_COEFFICIENT)


def pixel_accuracy(preds: Tensor,
                   labels: Tensor) -> Tensor:
    """
    Compute the pixel accuracy score given batches of predictions and labels.

    :param preds: batch of binary mask predictions
    :param labels: batch of binary mask labels
    :return: the pixel accuracy score for each pair in the batch
    """

    if not torch.is_tensor(preds):
        raise TypeError(f'Predictions type is not a torch.Tensor. Got {format(type(preds))}')
    if not torch.is_tensor(labels):
        raise TypeError(f'Labels type is not a torch.Tensor. Got {format(type(labels))}')
    if preds.dtype != torch.long:
        raise TypeError(f'Predictions should be tensors of type torch.long. Got {format(preds.dtype)}')
    if labels.dtype != torch.long:
        raise TypeError(f'Labels should be tensors of type torch.long. Got {format(labels.dtype)}')

    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    return (preds.eq(labels).long().sum((1, 2)) + SMOOTHING_COEFFICIENT) /\
           (labels.shape[1] * labels.shape[2] + SMOOTHING_COEFFICIENT)
