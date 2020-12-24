import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision.transforms import transforms


def display_images_and_masks(images: list,
                             masks: list = None,
                             labels: list = None,
                             columns: int = 5,
                             width: int = 20,
                             height: int = 8,
                             label_font_size: int = 9) -> None:
    """
    From https://www.kaggle.com/mariazorkaltseva/hubmap-seresnext50-unet-dice-loss
    Plot multiple images (max 15) in a grid-like structure. Masks are applied over images.

    :param images: list of NumPy arrays or PIL images
    :param masks: optional list of NumPy arrays or PIL masks
    :param labels: optional image labels
    :param columns: number of columns to display
    :param width: plot's width
    :param height: plot's height
    :param label_font_size: label's font size
    """

    max_images = 15

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]
        if masks is not None:
            masks = masks[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))

    if masks is not None:
        for i, (image, mask) in enumerate(zip(images, masks)):
            plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
            plt.imshow(image)
            plt.imshow(mask, cmap='coolwarm', alpha=0.5)

            if labels is not None:
                plt.title(labels[i], fontsize=label_font_size)
    else:
        for i, image in enumerate(images):
            plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
            plt.imshow(image)

            if labels is not None:
                plt.title(labels[i], fontsize=label_font_size)
    plt.show()


def visualize(**images: dict) -> None:
    """
    Plot images in one row. Remember to apply any operations to invert your transformations, such as denormalization,
    before passing to this function.

    :param images: dictionary of names and images as PIL images, NumPy arrays or PyTorch tensors.
    """

    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        # If input is a NumPy array, check if the color channels are in the last dimension
        if type(image) in [np.ndarray] and image.shape[0] in [1, 3]:
            image = image.reshape((image.shape[1], image.shape[2], -1))

        # If input is a PyTorch tensor containing floating point numbers transform to a PIL image using the
        # appropriate transformations
        if type(image) is torch.Tensor and image.dtype in [torch.float, torch.double]:
            # Expects a PyTorch tensor with values between 0 and 1
            image = transforms.ToPILImage()(image)
        elif type(image) is torch.Tensor and image.dtype in [torch.int, torch.long]:
            # Otherwise the image is considered to have values between 0 and 255
            image.numpy()

        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
