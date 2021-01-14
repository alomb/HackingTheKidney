import gc
import os
import zipfile
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff
import cv2

from utils import rle_decode


class ContextConfig:
    """
    Configuration class used to handle the creation of the context images.
    """
    def __int__(self,
                image_folder: str,
                mask_folder: str,
                ctx_size: int) -> None:
        """

        :param image_folder: path that will contain the images
        :param mask_folder: path that will contain the masks
        :param ctx_size: size of the context images
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.ctx_size = ctx_size


def create_context_target_dataset(image_size,
                                  input_path,
                                  image_folder,
                                  mask_folder,
                                  df_info,
                                  limit: int = 100,
                                  reduce_factor: int = 4,
                                  context_config: Optional[ContextConfig] = None) -> Tuple[float, float]:
    """
    Creates two zipped folders containing respectively images and related masks extracted as regions from the .tiff
    images and rle encodings in the given path.

    Adapted from https://www.kaggle.com/iafoss/256x256-images.

    :param image_size: size of the extracted regions
    :param input_path: path containing images and masks.
    (https://www.kaggle.com/c/hubmap-kidney-segmentation/data?select=train)
    :param image_folder: path that will contain the images
    :param mask_folder: path that will contain the masks
    :param df_info: dataframe containing information on the images/masks. Should be extracted from
    https://www.kaggle.com/c/hubmap-kidney-segmentation/data?select=train.csv
    :param limit: number of images to parse
    :param reduce_factor: reducing factor on the original image
    :param context_config: information about the context images, if None contexts are not extracted

    :return mean and std computed from the images
    """

    # Saturation blanking threshold
    s_th = 40
    # Threshold for the minimum number of pixels to keep an image
    p_th = 200 * image_size // 256

    mean_sum = []
    std_sum = []

    with zipfile.ZipFile(image_folder, 'w') as img_out, \
            zipfile.ZipFile(mask_folder, 'w') as mask_out:
        for index, encs in df_info.head(limit).iterrows():
            print(f'Process image {index}')
            # Read image and generate the mask
            img = tiff.imread(os.path.join(input_path, index + '.tiff'))
            if len(img.shape) == 5:
                img = np.transpose(img.squeeze(), (1, 2, 0))
            mask = rle_decode(encs, (img.shape[1], img.shape[0]))

            # Add padding to make the image dividable into tiles
            shape = img.shape
            pad0 = (reduce_factor * image_size - shape[0] % (reduce_factor * image_size)) % (reduce_factor * image_size)
            pad1 = (reduce_factor * image_size - shape[1] % (reduce_factor * image_size)) % (reduce_factor * image_size)
            img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                         constant_values=0)
            mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]],
                          constant_values=0)

            # Split image and mask into tiles using the reshape+transpose trick
            img = cv2.resize(img, (img.shape[1] // reduce_factor, img.shape[0] // reduce_factor),
                             interpolation=cv2.INTER_AREA)
            img = img.reshape(img.shape[0] // image_size, image_size, img.shape[1] // image_size, image_size, 3)

            # Number of columns
            cols = img.shape[2]

            img = img.transpose(0, 2, 1, 3, 4).reshape(-1, image_size, image_size, 3)

            mask = cv2.resize(mask, (mask.shape[1] // reduce_factor, mask.shape[0] // reduce_factor),
                              interpolation=cv2.INTER_NEAREST)
            mask = mask.reshape(mask.shape[0] // image_size, image_size, mask.shape[1] // image_size, image_size)
            mask = mask.transpose(0, 2, 1, 3).reshape(-1, image_size, image_size)

            # Columns and rows positions used then to extract contexts of the chosen images
            col = 0
            row = -1
            cols_ctx = []
            rows_ctx = []
            filenames_ctx = []

            # Write data
            for i, (im, m) in enumerate(zip(img, mask)):
                # Update positions
                if i % cols != 0:
                    col += 1
                else:
                    col = 0
                    row += 1

                # Remove black or gray images based on saturation check
                hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                if (s > s_th).sum() <= p_th or im.sum() <= p_th:
                    continue

                cols_ctx.append(col)
                rows_ctx.append(row)
                filenames_ctx.append(i)

                # Update sums used to compute mean and std of the dataset's images
                mean_sum.append((im / 255.0).reshape(-1, 3).mean(0))
                std_sum.append(((im / 255.0) ** 2).reshape(-1, 3).mean(0))

                # Write on memory
                im = cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f'{index}_{i}.png', im)
                m = cv2.imencode('.png', m)[1]
                mask_out.writestr(f'{index}_{i}.png', m)

            # Free space
            del img
            del mask
            gc.collect()

        if context_config is not None:
            with zipfile.ZipFile(context_config.image_folder, 'w') as img_out_ctx, \
                    zipfile.ZipFile(context_config.mask_folder, 'w') as mask_out_ctx:

                print("Creating context images...")
                # read image and generate the mask
                img_ctx = tiff.imread(os.path.join(input_path, index + '.tiff'))
                if len(img_ctx.shape) == 5:
                    img_ctx = np.transpose(img_ctx.squeeze(), (1, 2, 0))
                mask_ctx = rle_decode(encs, (img_ctx.shape[1], img_ctx.shape[0]))

                pad0_ctx = pad0 + (context_config.ctx_size * reduce_factor - image_size * reduce_factor)
                pad1_ctx = pad1 + (context_config.ctx_size * reduce_factor - image_size * reduce_factor)

                img_ctx = np.pad(img_ctx,
                                 [[pad0_ctx // 2, pad0_ctx - pad0_ctx // 2],
                                  [pad1_ctx // 2, pad1_ctx - pad1_ctx // 2],
                                  [0, 0]],
                                 constant_values=0)
                mask_ctx = np.pad(mask_ctx,
                                  [[pad0_ctx // 2, pad0_ctx - pad0_ctx // 2],
                                   [pad1_ctx // 2, pad1_ctx - pad1_ctx // 2]],
                                  constant_values=0)

                img_ctx = cv2.resize(img_ctx, (img_ctx.shape[1] // reduce_factor, img_ctx.shape[0] // reduce_factor),
                                     interpolation=cv2.INTER_AREA)

                mask_ctx = cv2.resize(mask_ctx,
                                      (mask_ctx.shape[1] // reduce_factor, mask_ctx.shape[0] // reduce_factor),
                                      interpolation=cv2.INTER_NEAREST)

                for i, (row, col) in enumerate(zip(rows_ctx, cols_ctx)):
                    # Cut region
                    # Vertical and horizontal start and end region delimiters
                    v_s = row * image_size
                    v_e = row * image_size + context_config.ctx_size
                    h_s = col * image_size
                    h_e = col * image_size + context_config.ctx_size

                    im_ctx = img_ctx[v_s: v_e, h_s: h_e]
                    m_ctx = mask_ctx[v_s: v_e, h_s: h_e]

                    # Write on memory
                    im_ctx = cv2.imencode('.png', cv2.cvtColor(im_ctx, cv2.COLOR_RGB2BGR))[1]
                    img_out_ctx.writestr(f'{index}_{filenames_ctx[i]}.png', im_ctx)
                    m_ctx = cv2.imencode('.png', m_ctx)[1]
                    mask_out_ctx.writestr(f'{index}_{filenames_ctx[i]}.png', m_ctx)

            del img_ctx
            del mask_ctx
            gc.collect()

    mean = np.mean(mean_sum)
    return mean, np.sqrt(np.mean(std_sum) - mean**2)
