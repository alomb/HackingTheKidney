import os
import random
from typing import Tuple, List, Dict, Optional, Union

import albumentations
import cv2
from PIL import Image
import numpy as np

from torchvision import transforms
import torch
from torch.utils.data import Dataset


def get_training_validation_sets(images_path: str,
                                 masks_path: str,
                                 validation_percentage: float,
                                 dataset_augmentations: Dict,
                                 device: str,
                                 mean: Optional[List[float]] = None,
                                 std: Optional[List[float]] = None) -> Tuple[Dataset, set, Dataset, set]:
    """
    Function used to build datasets and filenames of training and validation splits.

    :param images_path: path of the folder containing all the images
    :param masks_path: path of the folder containing all the masks
    :param validation_percentage: percentage of samples used to populate the validation split
    :param dataset_augmentations: transformations for the training and validation sets
    :param device the device used 'cpu' or 'cuda'
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
                                 device,
                                 mean,
                                 std,
                                 augmentations=dataset_augmentations['train'])

    validation_set = HuBMAPDataset(list(validation_images),
                                   images_path,
                                   masks_path,
                                   device,
                                   mean,
                                   std,
                                   augmentations=dataset_augmentations['val'])

    return training_set, training_images, validation_set, validation_images


def denormalize_images(images: torch.FloatTensor,
                       mean: List[float],
                       std: List[float]) -> torch.FloatTensor:
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
    - Apply custom augmentations
    - Convert to normalized PyTorch tensors if mean and std are passed
    - Its iterator returns image and its masks
    """

    def __init__(self,
                 images: List[str],
                 images_path: str,
                 masks_path: str,
                 device: str,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 augmentations: Optional[albumentations.Compose] = None):
        """
        If mean and standard deviations are not passed images are not transformed to tensors.

        :param images: list of sample names
        :param images_path: path of the folder containing all the images
        :param masks_path: path of the folder containing all the masks
        :param device the device used 'cpu' or 'cuda' where PyTorch tensors are passed
        :param mean: mean for each channel (RGB)
        :param std: standard deviation of each channel (RGB)
        :param augmentations: image augmentations operations
        """

        self.images = images
        self.image_path = images_path
        self.mask_path = masks_path
        self.augmentations = augmentations
        self.device = device

        self.mean = mean
        self.std = std

        # Transform the PIL image or NumPy array to a tensor with values between 0.0 and 1.0 and then normalize each
        # channel with the given mean and standard deviation.
        if mean is not None and std is not None:
            self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
        else:
            self.to_tensor = None

    def __len__(self) -> int:
        """
        :return: the dataset length
        """

        return len(self.images)

    def __getitem__(self, index: int) -> Union[Tuple[torch.FloatTensor, torch.LongTensor],
                                               Tuple[np.ndarray, np.ndarray]]:
        """
        Apply transformations and return the image and its mask

        :param index:
        :return: the index-th image and binary mask as PyTorch tensors (3, H, W) if mean and std are passed otherwise
        as Numpy arrays (H, W, 3)
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
        if self.to_tensor is not None:
            return self.to_tensor(img).to(self.device), torch.from_numpy(np.array(mask)).long().to(self.device)
        else:
            return img, np.array(mask)


class ContextualHuBMAPDataset:
    """
    Dataset containing two HuBMAPDataset for target and context images and masks (used in HookNet). It handles the
    transformations and the augmentations of the images and masks. The given datasets must return Numpy images.
    """

    def __init__(self,
                 target_dataset: HuBMAPDataset,
                 context_dataset: HuBMAPDataset,
                 device: str,
                 reduction: int,
                 mean: Optional[List[float]],
                 std: Optional[List[float]],
                 augmentations: Optional[albumentations.Compose] = None):
        """
        :param target_dataset: dataset containing images and masks for the target branch (same number of images of the
        context dataset)
        :param context_dataset: dataset containing images and masks for the context branch (same number of images of the
        target dataset)
        :param device the device used 'cpu' or 'cuda' where PyTorch tensors are passed
        :param reduction: a reduction factor applied to the contextual images
        :param mean: mean for each channel (RGB)
        :param std: standard deviation of each channel (RGB)
        :param augmentations: image augmentations operations. It is expected to have been declared passing
        **additional_targets = {'context_image': 'image', 'context_mask': 'image'}** to transform context and target
        images equally.
        """

        assert len(target_dataset.images) == len(context_dataset.images), \
            'Contextual and target dataset must have the same number of images and masks!'

        assert target_dataset.augmentations is None or context_dataset.augmentations is None, \
            'Contextual and target dataset must not apply independent augmentations! Images must be equally augmented.'

        assert target_dataset.to_tensor is None or context_dataset.to_tensor is None, \
            'Contextual and target dataset must not return PyTorch tensors but Numpy arrays in order to apply here ' \
            'the required transformations!'

        if augmentations is not None:
            assert 'context_mask' in augmentations.additional_targets and \
                   'context_image' in augmentations.additional_targets, "Augmentations is expected to have been " \
                                                                        "declared passing **additional_targets = " \
                                                                        "{'context_image': 'image', 'context_mask': " \
                                                                        "image'}** to transform context and target " \
                                                                        "images equally."

        self.target_dataset = target_dataset
        self.context_dataset = context_dataset
        self.device = device

        self.reduction = reduction
        self.augmentations = augmentations

        self.mean = mean
        self.std = std
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

    def __len__(self):
        """
        :return: the dataset length
        """

        return len(self.target_dataset.images)

    def __getitem__(self, index) -> Tuple[Tuple[torch.FloatTensor, torch.LongTensor],
                                          Tuple[torch.LongTensor, torch.LongTensor]]:
        """
        Apply transformations and return the images and their masks of both target and context dataset

        :param index:
        :return: target and context's index-th image and binary mask as tensors. (target_image, context_image,
        target_mask, context_mask)
        """

        target_image, target_mask = self.target_dataset[index]
        context_image, context_mask = self.context_dataset[index]

        # Apply augmentations
        if self.augmentations is not None:
            transformed = self.augmentations(image=target_image,
                                             mask=target_mask,
                                             context_image=context_image,
                                             context_mask=context_mask)

            target_image = transformed['image']
            target_mask = transformed['mask']
            context_image = transformed['context_image']
            context_mask = transformed['context_mask']

        context_resized = albumentations.Resize(context_image.shape[0] // self.reduction,
                                                context_image.shape[1] // self.reduction,
                                                interpolation=cv2.INTER_AREA, p=1)(image=context_image,
                                                                                   mask=context_mask)

        context_image, context_mask = context_resized['image'], context_resized['mask']
        # Transform to tensor
        return (self.to_tensor(target_image).to(self.device),
                self.to_tensor(context_image).to(self.device)), \
               (torch.from_numpy(np.array(target_mask)).long().to(self.device),
                torch.from_numpy(np.array(context_mask)).long().to(self.device))
