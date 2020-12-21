import os
import time
from datetime import datetime
from typing import Union

import numpy as np

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from evaluation.metrics import iou, pixel_accuracy, dice_coefficient
from preprocessing.dataset import HuBMAPDataset


class Statistics:
    """
    Class used to register and keep track of the training statistics.
    """

    def __init__(self,
                 epochs: int,
                 metrics: list[str],
                 accumulate: bool = False):
        """
        :param epochs: number of total epochs
        :param metrics: list of metrics names
        :param accumulate: if True maintains lists of metrics, otherwise sum values at each update
        """

        self.accumulate = accumulate

        if accumulate:
            self.stats = {epoch: {s: [] for s in metrics}
                          for epoch in range(1, epochs + 1)}
        else:
            self.stats = {epoch: {s: (None, None) for s in metrics}
                          for epoch in range(1, epochs + 1)}

    def update(self, epoch: int, stat: str, value: Union[int, float]) -> None:
        """
        :param epoch: current epoch
        :param stat: the stat to update
        :param value: the new value to be added
        """

        curr_value = self.stats[epoch][stat]

        if self.accumulate:
            curr_value.append(value)
        else:
            # Update value
            self.stats[epoch][stat][0] = (curr_value[0] + value) if curr_value[0] is not None else value

            # Update counter
            self.stats[epoch][stat][1] = (curr_value[1] + 1) if curr_value[1] is not None else 1

    def get_all_averages(self) -> dict:
        """
        :return: averaged statistics at each epoch
        """

        if self.accumulate:
            return {e: {s: np.mean(self.stats[e][s]) for s in self.stats[e]} for e in self.stats}
        else:
            return {e: {s: (self.stats[e][s][0] / self.stats[e][s][1]) for s in self.stats[e]} for e in self.stats}

    def get_averaged_stat(self, stat: str) -> list:
        """
        :param stat: the selected state
        :return: the state stats averaged at each epoch
        """

        if self.accumulate:
            return [np.mean(self.stats[e][stat]) for e in self.stats]
        else:
            return [self.stats[e][stat][0] / self.stats[e][stat][1] for e in self.stats]

    def __str__(self) -> str:
        return self.stats.__str__()


class Trainer:
    """
    Class used to instantiate and run a test.
    """

    def __init__(self,
                 model: Module,
                 threshold: float,
                 criterion: Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataset: HuBMAPDataset,
                 batch_size: int,
                 root_path: str,
                 device: str):
        """
        :param model: model to train
        :param threshold: minimum value used to threshold model outputs: predicted mask = output > threshold
        :param criterion: loss function
        :param optimizer: optimizer used during training
        :param training_dataset: custom dataset to retrieve images and masks
        :param batch_size: size of batches used to create a DataLoader
        :param root_path: the path of the root
        :param device: device used
        """

        self.model = model
        self.threshold = threshold
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_data_loader = DataLoader(training_dataset,
                                               batch_size=batch_size)
        self.root_path = root_path
        self.device = device

    def train(self,
              epochs: int,
              weights_dir: str = datetime.now().strftime("%d_%m_%y_%H_%M_%S"),
              verbose: bool = False) -> Statistics:
        """
        Train the model

        :param epochs: number of epochs used to train
        :param weights_dir: path of the directory from the root used to save weights
        :param verbose: if True print progress and info during training
        :return: statistics
        """

        # Set training mode
        self.model.train()

        # Initialize statistics
        stats = Statistics(epochs,
                           ['epoch_train_time',
                            'batch_train_time',
                            'loss',
                            'iou',
                            'dice_coefficient',
                            'pixel_accuracy'],
                           accumulate=True)

        # Create directory containing weights
        if weights_dir is not None and weights_dir != '':
            weights_dir = os.path.join(self.root_path, weights_dir)
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)

        epoch_start_time = time.time()
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"Training epoch {epoch}:")

            for images, masks in iter(self.training_data_loader):
                batch_start_time = time.time()

                images = images.to(self.device)
                masks = masks.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs['out'], masks)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                preds = (outputs['out'] > self.threshold).long()

                # Update stats
                stats.update(epoch, 'loss', loss)
                stats.update(epoch, 'iou', iou(preds, masks).mean())
                stats.update(epoch, 'dice_coefficient', dice_coefficient(preds, masks).mean())
                stats.update(epoch, 'pixel_accuracy', pixel_accuracy(preds, masks).mean())
                stats.update(epoch, 'batch_train_time', time.time() - batch_start_time)

            if weights_dir is not None and weights_dir != '':
                torch.save(self.model.state_dict(), os.path.join(weights_dir, f'weights_{epoch}.js'))

            stats.update(epoch, 'epoch_train_time', time.time() - epoch_start_time)
            if verbose:
                print(f"Epoch {epoch} ended in {stats.stats[epoch]['epoch_train_time'][-1]} seconds")
                print('Average training batch time', np.mean(stats.stats[epoch]['batch_train_time']), 'seconds')
                print('Average loss', np.mean(stats.stats[epoch]['loss']))
                print('Metrics:')
                print('Average IoU \t\t\t\t', np.mean(stats.stats[epoch]['iou']))
                print('Average dice coefficient \t', np.mean(stats.stats[epoch]['dice_coefficient']))
                print('Average pixel accuracy \t\t', np.mean(stats.stats[epoch]['pixel_accuracy']))
                print(f"{'-'*100}")

        return stats
