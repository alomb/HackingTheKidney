from typing import Tuple

import numpy as np

import torch


def rle_decode(mask_rle: str,
               shape: Tuple[int, int]) -> np.array:
    """
    Decode the RLE string.

    :param mask_rle: run-length as formatted string 'start1 length1 start2 length2 ...'
    :param shape: (height,width) of  array to return
    :return: numpy array where 1 is mask and 0 is background
    """

    # Get start and length pairs
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill starts-ends regions with 1
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def rle_encode(img: np.array) -> str:
    """
    Encode the mask in the RLE format.

    :param img: numpy array where 1 is mask and 0 is background
    :return: run length as formatted string
    """

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_encode_less_memory(img: np.array) -> str:
    """
    Avoid the OOM exception caused by the np.concatenate function.
    From https://www.kaggle.com/bguberfain/memory-aware-rle-encoding.
    This simplified method requires first and last pixel to be zero.

    :param img: numpy array where 1 is mask and 0 is background
    :return: run length as formatted string
    """

    pixels = img.T.flatten()
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def global_shift_mask(mask_pred: np.array,
                      y_shift: int,
                      x_shift: int) -> np.array:
    """
    Applies a global shift to a mask by padding one side and cropping from the other.
    Function taken from https://www.kaggle.com/tivfrvqhs5/global-mask-shift originally created to correct some images
    incorrect ground truth notations.

    :param mask_pred: the predicted mask
    :param y_shift: vertical shift
    :param x_shift: horizontal shift
    :return: the shifted image
    """

    maskpred3 = None

    if y_shift < 0 and x_shift >= 0:
        maskpred2 = np.pad(mask_pred, [(0, abs(y_shift)), (abs(x_shift), 0)], mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, :mask_pred.shape[1]]

    elif y_shift >= 0 and x_shift < 0:
        maskpred2 = np.pad(mask_pred, [(abs(y_shift), 0), (0, abs(x_shift))], mode='constant', constant_values=0)
        maskpred3 = maskpred2[:mask_pred.shape[0], abs(x_shift):]

    elif y_shift >= 0 and x_shift >= 0:
        maskpred2 = np.pad(mask_pred, [(abs(y_shift), 0), (abs(x_shift), 0)], mode='constant', constant_values=0)
        maskpred3 = maskpred2[:mask_pred.shape[0], :mask_pred.shape[1]]

    elif y_shift < 0 and x_shift < 0:
        maskpred2 = np.pad(mask_pred, [(0, abs(y_shift)), (0, abs(x_shift))], mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, abs(x_shift):]

    return maskpred3


def get_device_colab() -> str:
    """

    Check GPU's availability.

    :return: the device name
    """
    device = "cpu"
    if torch.cuda.is_available:
        print('All good, a GPU is available')
        device = torch.device("cuda:0")
    else:
        print('Please set GPU via Edit -> Notebook Settings.')

    return device


def set_deterministic_colab(seed: int,
                            set_deterministic: bool = False) -> None:
    """
    Set a deterministic behaviour.
    https://pytorch.org/docs/stable/notes/randomness.html

    :param seed: the seed to set for PyTorch and Numpy
    :param set_deterministic: if True configure PyTorch to use deterministic algorithms instead of nondeterministic
    ones where available, and to throw an error if an operation is known to be nondeterministic (and without a
    deterministic alternative).

    :raise RuntimeError: if the version used is not supported
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configure cuDNN to deterministically select an algorithm at each run
    # Set False to change this behaviour, performance may be impacted
    torch.backends.cudnn.benchmark = True
    # Configure cuDNN to choose deterministic algorithms
    torch.backends.cudnn.deterministic = True

    # Configure PyTorch to use deterministic algorithms
    if set_deterministic:
        version = torch.__version__.split('+')[0]
        major, minor, patch = version.split(".")
        major, minor, patch = int(major), int(minor), int(patch)

        if major >= 1 and minor >= 7:
            torch.set_deterministic(True)
        else:
            raise RuntimeError('PyTorch 1.7 or higher is required to use PyTorch deterministic algorithms.')
