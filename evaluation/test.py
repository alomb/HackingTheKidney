import gc
import glob
import math
import os
import pathlib
from typing import Tuple, Dict, OrderedDict, List

import cv2
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm

import torch
from torch.nn import Module

from preprocessing.dataset import denormalize_images
from utils import rle_encode_less_memory
from visualization.visualize_data import display_images_and_masks


def make_grid(shape: Tuple[int, int],
              window: int,
              min_overlap: int) -> np.array:
    """
    Computes the coordinates where images can be divided into smaller slices.

    :param shape: tuple containing horizontal and vertical image sizes
    :param window: slice side length
    :param min_overlap: minimum number of overlapping pixels which regulates how much the slices can be overlapped
    :return: a narray containing the slices saved as starting and ending points in the two axis
    """

    def get_starts_ends(length: int, knots: int) -> Tuple[np.array, np.array]:
        """
        Split into uniform parts a given length and return starting and ending points.

        :param length: total length to split
        :param knots: number of splits
        :return: two narrays containing indices where splits respectively start and end
        """

        # Obtain slices excluding the last index which indicates the termination
        starts = np.linspace(0, length, num=knots, endpoint=False, dtype=np.int64)
        starts[-1] = length - window
        ends = (starts + window).clip(0, length)
        return starts, ends

    x, y = shape
    # Compute number of splits
    nx = x // (window - min_overlap) + 1
    ny = y // (window - min_overlap) + 1

    x1, x2 = get_starts_ends(x, nx)
    y1, y2 = get_starts_ends(y, ny)

    # Save 2 coordinates for each slice
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]

    return slices.reshape(nx * ny, 4)


class Test:
    """
    Class responsible for testing a trained model.
    """

    def __init__(self,
                 model: Module,
                 threshold: float,
                 testing_directory_path: str,
                 window: int,
                 resize_factor: float,
                 min_overlap: int,
                 device: str):
        """
        :param model: model to test
        :param threshold: minimum value used to threshold model outputs: predicted mask = output > threshold
        :param testing_directory_path: path containing testing images
        :param window: window size to split the images
        :param resize_factor: resizing factor applied to the splits to downsample their size
        :param min_overlap: minimum overlap between the splits
        :param device: used device
        """

        self.model = model
        self.threshold = threshold
        self.testing_directory_path = pathlib.Path(testing_directory_path)
        self.window = window
        self.resize_factor = resize_factor
        self.min_overlap = min_overlap
        self.device = device

    def test(self,
             output_csv_file: str,
             mean: List[float],
             std: List[float],
             model_output_logits: bool = True,
             morphological_postprocessing: bool = True,
             verbose: bool = False,
             min_num_of_1_to_show_images: int = math.inf) -> Dict:
        """
        Evaluate the model on the images in the given directory.

        :param output_csv_file: path to the file where the encoded masks are saved (.csv format)
        :param mean: mean for each channel (RGB)
        :param std: standard deviation of each channel (RGB)
        :param model_output_logits: if true the model outputs logits which must be transformed into a probability
        :param verbose: if True prints details
        :param min_num_of_1_to_show_images: minimum number of 1 in the mask to decide whether to show it or not
        :return: dictionary containing for each image its rle encoded mask
        """

        self.model.eval()
        images_iterator = glob.glob1(self.testing_directory_path, "*.tiff")
        submission_dict = dict()

        transformation = T.Compose([T.ToPILImage(),
                                    T.Resize(int(self.window * self.resize_factor)),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])

        for i, filename in enumerate(images_iterator):
            if verbose:
                print(f'Evaluating image {filename} ({i}/{len(images_iterator)})')

            total_image = rasterio.open(os.path.join(self.testing_directory_path, filename))

            # Generate slices
            slices = make_grid(total_image.shape,
                               self.window,
                               self.min_overlap)

            # Total predicted mask with the size equal to the whole image
            total_pred_mask = np.zeros(total_image.shape, dtype=np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            if verbose:
                slices = tqdm(slices)

            for (x1, x2, y1, y2) in slices:
                # Read image slice
                image = total_image.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))

                # Move channels at the last dimension
                image = np.moveaxis(image, 0, -1)

                # Apply transformations (size rescaling, 0-1 rescaling, normalization)
                image = transformation(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Feed the model
                    output = self.model(image)

                    # Support torchvision.models
                    if type(output) is OrderedDict:
                        output = output['out']

                    # Convert logits
                    if model_output_logits:
                        output.sigmoid_()

                    # Move to NumPy and rescale it according to the resize factor
                    pred = output.squeeze().cpu().numpy()
                    final_pred = (cv2.resize(pred, (self.window, self.window)) > self.threshold).astype(np.uint8)

                    # Apply morphological transformations
                    if morphological_postprocessing:
                        final_pred = cv2.morphologyEx(final_pred, cv2.MORPH_CLOSE, kernel)
                        final_pred = cv2.morphologyEx(final_pred, cv2.MORPH_OPEN, kernel)

                    # Threshold and obtain final predictions
                    total_pred_mask[x1:x2, y1:y2] = final_pred

                    if final_pred.sum() >= min_num_of_1_to_show_images:
                        image = image.detach().cpu()

                        denormalized_image = T.ToPILImage()(denormalize_images(image.squeeze(), mean, std))

                        display_images_and_masks(images=[denormalized_image] * 3,
                                                 masks=[pred,
                                                        cv2.resize(final_pred, (256, 256))])

            submission_dict[i] = {'id': filename.replace('.tiff', ''),
                                  'predicted': rle_encode_less_memory(total_pred_mask)}
            print(submission_dict)
            # Free some memory
            del total_pred_mask
            gc.collect()

        submission = pd.DataFrame.from_dict(submission_dict, orient='index')
        submission.to_csv(output_csv_file, index=False)
        return submission_dict
