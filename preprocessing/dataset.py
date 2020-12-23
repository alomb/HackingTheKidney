import os
import random
from typing import Tuple

import albumentations
from PIL import Image
import numpy as np

from torchvision import transforms
import torch
from torch.utils.data import Dataset

from visualization.visualize_data import visualize


def get_training_validation_sets(images_path: str,
                                 masks_path: str,
                                 validation_percentage: float,
                                 dataset_augmentations: dict,
                                 mean: list[float],
                                 std: list[float]) -> Tuple[Dataset, set, Dataset, set]:
    """
    Function used to build datasets and filenames of training and validation splits.

    :param images_path: path of the folder containing all the images
    :param masks_path: path of the folder containing all the masks
    :param validation_percentage: percentage of samples used to populate the validation split
    :param dataset_augmentations: transformations for the training and validation sets
    :param mean: mean for each channel (RGB)
    :param std: standard deviation of each channel (RGB)

    :return: a torch.utils.data.Dataset and list of filenames for both training and validation splits
    """

    # Computes filenames/images identifiers
    training_images = set(os.listdir(images_path))

    validation_images = set(random.sample(os.listdir(images_path),
                                          int(validation_percentage * len(training_images))))
    training_images -= validation_images

    # Creates datasets
    training_set = HuBMAPDataset(list(training_images),
                                 images_path,
                                 masks_path,
                                 dataset_augmentations['train'],
                                 mean,
                                 std)

    validation_set = HuBMAPDataset(list(validation_images),
                                   images_path,
                                   masks_path,
                                   dataset_augmentations['val'],
                                   mean,
                                   std)

    return training_set, training_images, validation_set, validation_images


def denormalize_images(images: torch.FloatTensor,
                       mean: list[float],
                       std: list[float]) -> None:
    """
    Denormalize images.

    :param images: a image/batch of images
    :param mean: mean previously applied to normalize each channel (RGB)
    :param std: standard deviation previously applied to normalize each channel (RGB)
    :return: denormalized image/batch of images
    """
    denormalize_transformation = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                          std=[1 / std[0], 1 / std[1], 1 / std[2]]),
                                                     transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]],
                                                                          std=[1., 1., 1.])])
    return denormalize_transformation(images)


class HuBMAPDataset(Dataset):
    """
    The custom dataset.
    """

    def __init__(self,
                 images: list[str],
                 images_path: str,
                 masks_path: str,
                 augmentations: albumentations.Compose,
                 mean: list[float],
                 std: list[float]):
        """
        :param images: list of sample names
        :param images_path: path of the folder containing all the images
        :param masks_path: path of the folder containing all the masks
        :param augmentations: image augmentations operations
        :param mean: mean for each channel (RGB)
        :param std: standard deviation of each channel (RGB)
        """

        self.images = images
        self.image_path = images_path
        self.mask_path = masks_path
        self.augmentations = augmentations
        # Transform the PIL image or NumPy array to a tensor with values between 0.0 and 1.0 and then normalize each
        # channel with the given mean and standard deviation.
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        """
        :return: the dataset length
        """

        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Apply transformations and return the image and its mask

        :param index:
        :return: the index-th image and binary mask as tensors
        """

        filename = self.images[index]
        # A HxWx3 numpy array
        img = np.array(Image.open(os.path.join(self.image_path, filename)).convert('RGB'))
        # A HxW numpy array
        mask = np.array(Image.open(os.path.join(self.mask_path, filename)))

        # Apply augmentations
        if self.augmentations is not None:
            transformed = self.augmentations(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # Transform to tensor
        return self.to_tensor(img), torch.from_numpy(np.array(mask)).long()
