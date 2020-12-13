import numpy as np


def rle_decode(mask_rle: str,
               shape: tuple):
    """

    Decode the RLE string.

    :param mask_rle: run-length as formatted string 'start1 length1 start2 length2 ...'
    :param shape: (height,width) of  array to return
    :return: numpy array where 1 is mask and 0 is background
    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img: np.array):
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


def rle_encode_less_memory(img: np.array):
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
