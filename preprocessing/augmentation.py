import albumentations

import numpy as np
from albumentations import RandomRotate90
import albumentations.augmentations.functional as F


class Rotate270(RandomRotate90):
    """
    Rotate the input by 270 degrees.
    """

    def __init__(self):
        super().__init__()
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


def get_dihedral_transformations(probability: int = 1) -> albumentations.OneOf:
    """
    A dihedral group is the group of symmetries of a regular polygon, which includes rotations and reflections. These
    transformations do not destruct information and can be useful when images' direction do not have any meaning.
    For example a road sign may have a meaningful direction, changing it may change its class, medical or aerial imagery
    usually do not have such meaning.

    8 transformations:
    Rotates by 0°
    Rotates by 90°
    Rotates by 180°
    Rotates by 270°
    Flips vertically
    Flips horizontally
    Reflect on the semi-major axis
    Reflect on the semi-minor axis

    :param probability: weight associated to each transformation
    :return: a random dihedral transformation. It should be put inside an albumentations.Compose.
    """
    return albumentations.OneOf([albumentations.HorizontalFlip(p=probability),
                                 albumentations.VerticalFlip(p=probability),
                                 albumentations.RandomRotate90(p=probability),
                                 albumentations.Transpose(p=probability),
                                 albumentations.Sequential([Rotate270(),
                                                            albumentations.VerticalFlip(p=1)], p=probability)], 1)
