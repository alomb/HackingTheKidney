import gc
import glob
import os
import pathlib
from typing import Tuple

import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import Module

from utils import rle_encode_less_memory


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
    Class responsible of testing a trained model.
    """

    def __init__(self,
                 model: Module,
                 threshold: float,
                 testing_directory_path: str,
                 window: int,
                 min_overlap: int,
                 device: str):
        """
        :param model: model to test
        :param threshold: minimum value used to threshold model outputs: predicted mask = output > threshold
        :param testing_directory_path: path containing testing images
        :param window: window size to split the images
        :param min_overlap: minimum overlap between the splits
        :param device: used device
        """

        self.model = model
        self.threshold = threshold
        self.testing_directory_path = pathlib.Path(testing_directory_path)
        self.window = window
        self.min_overlap = min_overlap
        self.device = device

    def test(self, output_csv_file: str, verbose: bool = False) -> dict:
        """
        Evaluate the model on the images in the given directory.

        :param output_csv_file: path to the file where the encoded masks are saved (.csv format)
        :param verbose: if True prints details
        :return: dictionary containing for each image its rle encoded mask
        """

        self.model.eval()
        images_iterator = glob.glob1(self.testing_directory_path, "*.tiff")
        submission_dict = dict()

        for i, filename in enumerate(images_iterator):
            if verbose:
                print(f'Evaluating image {filename} ({i}/{len(images_iterator)})')

            whole_image = rasterio.open(os.path.join(self.testing_directory_path, filename))

            # Generate slices
            slices = make_grid(whole_image.shape,
                                      self.window,
                                      self.min_overlap)

            # Total predicted mask with the size equal to the whole image
            preds = np.zeros(whole_image.shape, dtype=np.uint8)

            for (x1, x2, y1, y2) in tqdm(slices):
                # Read image slice
                image = whole_image.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))

                # Move channels at the first dimension
                # image = np.moveaxis(image, 0, -1)
                with torch.no_grad():
                    image = torch.from_numpy(image).to(torch.float32).unsqueeze(0).to(self.device)
                    output = self.model(image)['out']

                    preds[x1:x2, y1:y2] = (output > self.threshold).long()

            submission_dict[i] = {'id': filename.stem, 'predicted': rle_encode_less_memory(preds)}
            # Free some memory
            del preds
            gc.collect()

        submission = pd.DataFrame.from_dict(submission_dict, orient='index')
        submission.to_csv(output_csv_file, index=False)
        return submission_dict
