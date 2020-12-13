import os

import matplotlib

from preprocessing.dataset import get_training_validation_sets

matplotlib.use('WebAgg')

# Download unzip and put in the data folder this https://www.kaggle.com/iafoss/hubmap-256x256
images_path = os.path.join('data', '256x256', 'train')
masks_path = os.path.join('data', '256x256', 'masks')

training_set, training_images, validation_set, validation_images = get_training_validation_sets(images_path,
                                                                                                masks_path,
                                                                                                0.0,
                                                                                                {'train': None,
                                                                                                 'val': None})
