from collections import OrderedDict
from typing import List

import os
import sys
import numpy as np
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

from utils import convert_to_tensors
from preprocess.dataset import denormalize_images
from visualization.visualize_data import display_images_and_masks


tqdm = partial(tqdm, position=0, leave=True, file=sys.stdout)


def show_predictions(model: nn.Module,
                     images_path: str,
                     masks_path: str,
                     image_size: int,
                     mean: List[float],
                     std: List[float],
                     device: str = 'cpu',
                     model_output_logits: bool = True,
                     images_to_show: int = 10,
                     threshold: float = 0.5,
                     min_wrong_glom_percentage: float = 0,
                     max_wrong_glom_percentage: float = 100,
                     verbose: bool = False) -> None:
    """

    :param model: trained model
    :param images_path: path of the folder containing all the images
    :param masks_path: path of the folder containing all the masks
    :param image_size: size of the images and masks in the dataset
    :param mean: to denormalize the images
    :param std: to denormalize the images
    :param device: the device which performs the operation
    :param model_output_logits: if True apply the sigmoid
    :param images_to_show: number of images to show before quitting
    :param threshold: threshold applied to the predicted mask
    :param min_wrong_glom_percentage: minimum percentage of wrong foreground pixels to show the image
    :param max_wrong_glom_percentage: maximum percentage of wrong foreground pixels to show the image
    :param verbose: if True print error percentages

    """
    model.eval()
    shown_images = 0

    dataset = os.listdir(images_path)

    data_iter = tqdm(dataset) if verbose else dataset

    for filename in data_iter:
        with torch.no_grad():
            image, mask = convert_to_tensors(
                img_path=os.path.join(images_path, filename),
                mask_path=os.path.join(masks_path, filename),
                mean=mean,
                std=std,
                device=device
            )
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

            if min_wrong_glom_percentage <= wrong_fg_percentage <= max_wrong_glom_percentage:
                if verbose:
                    print(f'\n{filename}: {100 - wrong_fg_percentage}% correct')
                    # print('Wrong background percentage:', wrong_bg_percentage, '\n')

                denormalized_image = T.ToPILImage()(denormalize_images(image.squeeze(), mean, std))

                display_images_and_masks(images=[denormalized_image] * 2,
                                         masks=[pred, mask],
                                         labels=['Prediction', 'Ground Truth'])

                shown_images += 1
                if shown_images == images_to_show:
                    break
