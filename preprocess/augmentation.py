import albumentations

import numpy as np
import cv2
from typing import Dict

from albumentations import RandomRotate90
import albumentations.augmentations.functional as F


class Rotate270(RandomRotate90):
    """
    Rotate the input by 270 degrees. Dummy class to imitate the same behavior of the RandomRotate90.
    """

    def __init__(self, p: float = 0.5):
        super(Rotate270, self).__init__(p=p)
        self.factor = 3

    def apply(self, img: np.array, factor: int = 0, **params):
        """
        :param img: image to rotate
        :param factor: (int): number of times the input will be rotated by 90 degrees.

        :return rotated image
        """
        return np.ascontiguousarray(np.rot90(img, self.factor))

    def get_params(self):
        return {"factor": self.factor}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, self.factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, self.factor, **params)

    def get_transform_init_args_names(self):
        return ()


def get_dihedral_transformations(probability: float = 1) -> albumentations.OneOf:
    """
    A dihedral group is the group of symmetries of a regular polygon, which includes rotations and reflections. These
    transformations do not destruct information and can be useful when images' direction do not have any meaning.
    For example a road sign may have a meaningful direction, changing it may change its class, medical or aerial imagery
    usually do not have such meaning.

    8 possible transformations:
    Rotates by 0°
    Rotates by 90°
    Rotates by 180°
    Rotates by 270°
    Flips vertically
    Flips horizontally
    Reflect on the semi-major axis
    Reflect on the semi-minor axis

    :param probability: associated to this transformation
    :return: an albumentations.OneOf which will choose a random dihedral transformation. It should be put inside an
    albumentations.Compose.
    """

    return albumentations.OneOf([albumentations.HorizontalFlip(p=1),
                                 albumentations.VerticalFlip(p=1),
                                 albumentations.RandomRotate90(p=1),
                                 albumentations.Transpose(p=1),
                                 albumentations.Sequential([Rotate270(p=1),
                                                            albumentations.VerticalFlip(p=1)], p=1)], probability)


def get_augmentations(dihedral_p: float = 0.5,
                      distortion_p: float = 0.33,
                      color_p: float = 0.25,
                      resize_size: int = None,
                      targets: Dict[str, str] = None) -> albumentations.Compose:
    """
    Compose:
    - dihedral augmentations
    - distortion augmentationsFinally

    :param dihedral_p: probability to apply the dihedral augmentation
    :param distortion_p: probability to apply the distortion augmentation
    :param color_p: probability to apply the color augmentation
    :param resize_size: size used to resize both train and test images if None resizing is not applied
    :param targets: Dict with keys - new target name, values - old target name.
                    To be used in order to transform multiple images with the same augmentation
    :return: a composition of augmentations
    """

    if resize_size is not None:
        resizing = [albumentations.Resize(resize_size, resize_size, interpolation=cv2.INTER_AREA, p=1)]
    else:
        resizing = []
    return albumentations.Compose([
        get_dihedral_transformations(dihedral_p),
        albumentations.OneOf([
            albumentations.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            albumentations.GridDistortion(p=1),
            albumentations.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=distortion_p),
        albumentations.OneOf([
            albumentations.HueSaturationValue(10, 15, 10, p=1),
            albumentations.CLAHE(clip_limit=2, p=1),
            albumentations.RandomBrightnessContrast(p=1),
            albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1)
        ], p=color_p)
    ] + resizing, additional_targets=targets)

