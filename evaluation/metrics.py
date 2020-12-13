import torch

IoU_smoothing_coeff = 1e-6


def iou(preds: torch.Tensor,
        labels: torch.Tensor):
    """
    Compute IoU scores given batches of predictions and labels

    :param preds: batch of mask predictions
    :param labels: batch of mask labels
    :return: the IoU score for each pair
    """

    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    intersections = (preds & labels).long().sum((1, 2))
    unions = (preds | labels).long().sum((1, 2))

    return (intersections + IoU_smoothing_coeff) / (unions + IoU_smoothing_coeff)


def pixel_accuracy(preds: torch.Tensor,
                   labels: torch.Tensor):
    """
    Compute the pixel accuracy score given batches of predictions and labels

    :param preds: batch of mask predictions
    :param labels: batch of mask labels
    :return: the pixel accuracy score
    """
    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    return preds.eq(labels).long().sum() / labels.numel()
