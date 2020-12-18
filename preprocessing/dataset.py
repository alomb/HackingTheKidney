import os
import random
from typing import Tuple

import torchvision
from PIL import Image

import torch
import numpy as np
from torch import Tensor

from torch.utils.data import Dataset


def get_training_validation_sets(images_path: str,
                                 masks_path: str,
                                 validation_percentage: float,
                                 dataset_transforms: dict) -> Tuple[Dataset, set, Dataset, set]:
    """

    :param images_path: path of the folder containing all the images
    :param masks_path: path of the folder containing all the masks
    :param validation_percentage: percentage of samples used to populate the validation split
    :param dataset_transforms: transformations for the training and validation sets
    :return: a torch.utils.data.Dataset and list of filenames for both training and validation splits
    """
    training_images = set(os.listdir(images_path))

    validation_images = set(random.sample(os.listdir(images_path),
                                          int(validation_percentage * len(training_images))))
    training_images -= validation_images

    training_set = HuBMAPDataset(list(training_images),
                                 images_path,
                                 masks_path,
                                 dataset_transforms['train'])

    validation_set = HuBMAPDataset(list(validation_images),
                                   images_path,
                                   masks_path,
                                   dataset_transforms['val'])

    return training_set, training_images, validation_set, validation_images


class HuBMAPDataset(Dataset):
    def __init__(self,
                 images: list,
                 images_path: str,
                 masks_path: str,
                 transforms: torchvision.transforms):
        """

        :param images: list of sample names
        :param images_path: path of the folder containing all the images
        :param masks_path: path of the folder containing all the masks
        :param transforms: image transformation operations
        """
        self.images = images
        self.image_path = images_path
        self.mask_path = masks_path
        self.transforms = transforms

    def __len__(self) -> int:
        """

        :return: the dataset length
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Apply transformations and return the image and its mask

        :param index:
        :return: the index-th element
        """
        filename = self.images[index]
        img = Image.open(os.path.join(self.image_path, filename)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, filename))

        if self.transforms is not None:
            img = self.transforms(img)

        return img, torch.from_numpy(np.array(mask)).long()
