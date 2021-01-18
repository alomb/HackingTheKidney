from collections import OrderedDict
from typing import List
import sys

import numpy as np
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

from preprocess.dataset import HuBMAPDataset, denormalize_images
from visualization.visualize_data import display_images_and_masks

# Change the tqdm function to avoid new lines and formatting errors
tqdm = partial(tqdm, position=0, leave=True, file=sys.stdout)


def show_predictions(model: nn.Module,
                     dataset: HuBMAPDataset,
                     image_size: int,
                     mean: List[float],
                     std: List[float],
                     model_output_logits: bool = True,
                     images_to_show: int = 10,
                     threshold: float = 0.5,
                     min_wrong_fg_percentage: float = 0,
                     min_wrong_bg_percentage: float = 0,
                     max_wrong_fg_percentage: float = 100,
                     max_wrong_bg_percentage: float = 100,
                     verbose=False) -> None:
    """

    :param model: trained model
    :param dataset: HuBMAP dataset
    :param image_size: size of the images and masks in the dataset
    :param mean: to denormalize the images
    :param std: to denormalize the images
    :param model_output_logits: if True apply the sigmoid
    :param images_to_show: number of images to show before quitting
    :param threshold: threshold applied to the predicted mask
    :param min_wrong_fg_percentage: minimum percentage of wrong foreground pixels to show the image
    :param min_wrong_bg_percentage: minimum percentage of wrong background pixels to show the image
    :param max_wrong_fg_percentage: maximum percentage of wrong foreground pixels to show the image
    :param max_wrong_bg_percentage: maximum percentage of wrong background pixels to show the image
    :param verbose: if True print error percentages

    """
    model.eval()
    shown_images = 0

    for image, mask in tqdm(dataset):
        with torch.no_grad():
            # Feed the model
            output = model(image.unsqueeze(0))

            # Support torchvision.models
            if type(output) is OrderedDict:
                output = output['out']

            # Convert logits
            if model_output_logits:
                output.sigmoid_()

            # Move to NumPy and rescale the mask according to the original size
            pred = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8)
            mask = mask.cpu().numpy()

            # Count wrong fg pixels
            wrong_fg_percentage = (((pred == 0) & (mask == 1)).sum() * 100) / (image_size ** 2)
            # Count wrong bg pixels
            wrong_bg_percentage = (((pred == 1) & (mask == 0)).sum() * 100) / (image_size ** 2)

            if (wrong_fg_percentage >= min_wrong_fg_percentage
                and (100 - wrong_fg_percentage) <= max_wrong_fg_percentage) \
                    or (wrong_bg_percentage >= min_wrong_bg_percentage
                        and (100 - wrong_bg_percentage) <= max_wrong_bg_percentage):
                if verbose:
                    print('\nWrong fg percentage:', wrong_fg_percentage)
                    print('Wrong bg percentage:', wrong_bg_percentage, '\n')

                denormalized_image = T.ToPILImage()(denormalize_images(image.squeeze(), mean, std))

                display_images_and_masks(images=[denormalized_image] * 2,
                                         masks=[pred, mask],
                                         labels=['prediction', 'ground truth'])

                shown_images += 1
                if shown_images == images_to_show:
                    break
