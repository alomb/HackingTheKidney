import torch


# smoothing value to avoid undefined and undetermined operations during the computations
smoothing_coeff = 1e-6


def iou(preds: torch.Tensor,
        labels: torch.Tensor):
    """
    Compute IoU scores given batches of predictions and labels

    :param preds: batch of mask predictions
    :param labels: batch of mask labels
    :return: the IoU score for each pair in the batch
    """

    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    intersections = (preds & labels).long().sum((1, 2))
    unions = (preds | labels).long().sum((1, 2))

    return (intersections + smoothing_coeff) / (unions + smoothing_coeff)


def dice_loss(preds: torch.Tensor,
              labels: torch.Tensor):
    """
    Compute the dice loss given batches of predictions and labels

    :param preds: batch of mask predictions
    :param labels: batch of mask labels
    :return: the dice loss for each pair in the batch
    """
    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    return (2. * preds.eq(labels).long().sum() + smoothing_coeff) / (labels.sum() + preds.sum() + smoothing_coeff)


def pixel_accuracy(preds: torch.Tensor,
                   labels: torch.Tensor):
    """
    Compute the pixel accuracy score given batches of predictions and labels

    :param preds: batch of mask predictions
    :param labels: batch of mask labels
    :return: the pixel accuracy score for each pair in the batch
    """
    # If the outputs are from a neural network they might have 4 dimensions (batch_size, 1, width, height)
    if preds.ndim == 4:
        preds = preds.squeeze(1)

    return (preds.eq(labels).long().sum() + smoothing_coeff) / (labels.numel() + smoothing_coeff)
