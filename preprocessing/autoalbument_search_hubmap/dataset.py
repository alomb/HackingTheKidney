import os
from typing import Tuple

from PIL import Image
import numpy as np

import torch.utils.data


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed
        print("Ciao", os.getcwd())
        self.images_path = os.path.join('..', '..', '..', '..', '..', 'data', '256x256', 'train')
        self.masks_path = os.path.join('..', '..', '..', '..', '..', 'data', '256x256', 'masks')
        self.images_id = os.listdir(self.images_path)

    def __len__(self) -> int:
        # Replace `...` with the actual implementation
        return len(self.images_id)

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.

        filename = self.images_id[index]
        image = np.array(Image.open(os.path.join(self.images_path, filename)).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.masks_path, filename))).astype(np.float)

        return image, np.expand_dims(mask, axis=-1)
