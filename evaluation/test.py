import gc
import glob
import math
import os
import pathlib
from typing import Tuple, Dict, OrderedDict, List

import albumentations
import cv2
import rasterio
from albumentations import HorizontalFlip, VerticalFlip, Rotate, Lambda, Transpose
from rasterio.windows import Window
import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm

import torch
from torch.nn import Module

from preprocess.dataset import denormalize_images
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

        new_size = int(self.window * self.resize_factor)

        transformation = T.Compose([T.ToPILImage(),
                                    T.Resize(new_size),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

                    # Move to NumPy and rescale the mask according to the original size
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

                        """
                        visualize(mask=pred)

                        plt.hist(pred.ravel(),256,[0,256])
                        plt.show()
                        print(pred.mean())
                        """

                        denormalized_image = T.ToPILImage()(denormalize_images(image.squeeze(), mean, std))

                        display_images_and_masks(images=[denormalized_image] * 2,
                                                 masks=[pred,
                                                        cv2.resize(final_pred, (new_size, new_size))])

            submission_dict[i] = {'id': filename.replace('.tiff', ''),
                                  'predicted': rle_encode_less_memory(total_pred_mask)}
            print(submission_dict)
            # Free some memory
            del total_pred_mask
            gc.collect()

        submission = pd.DataFrame.from_dict(submission_dict, orient='index')
        submission.to_csv(output_csv_file, index=False)
        return submission_dict

    def test_tta(self,
                 output_csv_file: str,
                 mean: List[float],
                 std: List[float],
                 model_output_logits: bool = True,
                 morphological_postprocessing: bool = True,
                 verbose: bool = False,
                 min_num_of_1_to_show_images: int = math.inf) -> Dict:
        """
        Evaluate the model on the images in the given directory using Test Time Augmentation.

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

        new_size = int(self.window * self.resize_factor)

        transformation = T.Compose([T.ToPILImage(),
                                    T.Resize(new_size),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])

        # TTA transformations
        horizontal_flip = HorizontalFlip(p=1)
        vertical_flip = VerticalFlip(p=1)

        # List of augmentations for TTA
        tta_augs = [Rotate(limit=(0, 0), p=1),
                    horizontal_flip,
                    vertical_flip,
                    Rotate(limit=(90, 90), p=1),
                    Rotate(limit=(180, 180), p=1),
                    Rotate(limit=(270, 270), p=1),
                    Transpose(p=1),
                    albumentations.Sequential([Rotate(limit=(270, 270), p=1),
                                               albumentations.VerticalFlip(p=1)], p=1)]

        # List of de-augmentations corresponding to the above list
        tta_deaugs = [None,
                      horizontal_flip,
                      vertical_flip,
                      Rotate(limit=(-90, -90), p=1),
                      Rotate(limit=(-180, -180), p=1),
                      Rotate(limit=(-270, -270), p=1),
                      Transpose(p=1),
                      albumentations.Sequential([albumentations.VerticalFlip(p=1),
                                                 Rotate(limit=(-270, -270), p=1)], p=1)]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

            if verbose:
                slices = tqdm(slices)

            for (x1, x2, y1, y2) in slices:
                # Read image slice
                image = total_image.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))

                # Move channels at the last dimension
                image = np.moveaxis(image, 0, -1)

                # Initialize tta prediction
                tta_pred = None

                for tta_index, tta_aug in enumerate(tta_augs):
                    # Apply TTA
                    tta_image = tta_aug(image=image)['image']

                    # Apply transformations (size rescaling, 0-1 rescaling, normalization)
                    tta_image = transformation(tta_image).unsqueeze(0).to(self.device)

                    # Feed the model
                    with torch.no_grad():
                        output = self.model(tta_image)

                        # Support torchvision.models
                        if type(output) is OrderedDict:
                            output = output['out']

                        # Convert logits
                        if model_output_logits:
                            output.sigmoid_()

                        # Move to NumPy
                        output = output.squeeze().cpu().numpy()

                        # De-augment mask
                        if tta_deaugs[tta_index] is not None:
                            output = tta_deaugs[tta_index](image=image,
                                                           mask=output)['mask']

                        # Rescale the mask according to the original size
                        output = cv2.resize(output, (self.window, self.window))

                        # Update the TTA sum
                        if tta_pred is None:
                            tta_pred = output
                        else:
                            tta_pred += output

                # Compute the average of all TTA predictions
                final_pred = tta_pred / len(tta_augs)

                # Threshold and obtain final predictions
                total_pred_mask[x1:x2, y1:y2] = (final_pred > self.threshold).astype(np.uint8)

                # Apply morphological transformations
                if morphological_postprocessing:
                    final_pred = cv2.morphologyEx(final_pred, cv2.MORPH_CLOSE, kernel)
                    final_pred = cv2.morphologyEx(final_pred, cv2.MORPH_OPEN, kernel)

                if (final_pred > self.threshold).sum() >= min_num_of_1_to_show_images:
                    display_images_and_masks(images=[cv2.resize(image, (new_size, new_size))] * 2,
                                             masks=[cv2.resize(final_pred, (new_size, new_size)),
                                                    cv2.resize((final_pred > self.threshold).astype(np.uint8),
                                                               (new_size, new_size))])

            submission_dict[i] = {'id': filename.replace('.tiff', ''),
                                  'predicted': rle_encode_less_memory(total_pred_mask)}
            print(submission_dict)
            # Free some memory
            del total_pred_mask
            gc.collect()

        submission = pd.DataFrame.from_dict(submission_dict, orient='index')
        submission.to_csv(output_csv_file, index=False)
        return submission_dict
