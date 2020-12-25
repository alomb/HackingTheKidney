import os
import time
import pickle
from datetime import datetime
from collections import OrderedDict
from typing import Union, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from evaluation.metrics import iou, pixel_accuracy, dice_coefficient
from preprocessing.dataset import HuBMAPDataset


class Statistics:
    """
    Class used to register and keep track of the training and evaluating statistics.
    """

    def __init__(self,
                 epochs: int,
                 metrics: List[str],
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

    def get_all_averages(self) -> Dict:
        """
        :return: averaged statistics at each epoch
        """

        if self.accumulate:
            return {e: {s: np.mean(self.stats[e][s]) for s in self.stats[e]} for e in self.stats}
        else:
            return {e: {s: (self.stats[e][s][0] / self.stats[e][s][1]) for s in self.stats[e]} for e in self.stats}

    def get_averaged_stat(self, stat: str) -> List:
        """
        :param stat: the selected state
        :return: the state stats averaged at each epoch
        """

        if self.accumulate:
            return [np.mean(self.stats[e][stat]) for e in self.stats]
        else:
            return [self.stats[e][stat][0] / self.stats[e][stat][1] for e in self.stats]

    def save(self, path: str) -> None:
        """
        Save the stats using pickle.

        :param path: where to save the stats
        """
        with open(path, 'wb') as handle:
            pickle.dump(self.stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __str__(self) -> str:
        """

        :return: string used to print the object
        """
        return pd.DataFrame.from_dict(self.stats,
                                      orient='index').to_string()


class Trainer:
    """
    Class used to instantiate and run a test.
    """

    def __init__(self,
                 model: Module,
                 threshold: float,
                 criterion: Module,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int,
                 device: str,
                 root_path: str,
                 training_dataset: HuBMAPDataset,
                 validation_dataset: Optional[HuBMAPDataset] = None):
        """
        :param model: model to train
        :param threshold: minimum value used to threshold model outputs: predicted mask = output > threshold
        :param criterion: loss function
        :param optimizer: optimizer used during training
        :param batch_size: size of batches used to create a DataLoader
        :param device: device used
        :param root_path: the path of the root
        :param training_dataset: custom dataset to retrieve images and masks for training
        :param validation_dataset: optional custom dataset to retrieve images and masks for validation
        """

        # TODO in the future consider avoiding copy and pass the model
        self.model = model
        self.threshold = threshold
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.training_data_loader = DataLoader(training_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
        self.validation_data_loader = DataLoader(validation_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        self.root_path = root_path
        self.device = device

    def evaluate(self,
                 epoch: int,
                 stats: Statistics,
                 verbose: bool = True,
                 limit: int = 2) -> None:
        """
        Method used to evaluate the model

        :param epoch: current epoch
        :param stats: statistics tracker
        :param verbose: if True print progress and info during evaluation
        :param limit TODO remove it
        """

        with torch.no_grad():
            epoch_start_time = time.time()

            if verbose:
                data_stream = tqdm(iter(self.validation_data_loader))
            else:
                data_stream = iter(self.validation_data_loader)

            # TODO REMOVE
            l = limit
            for images, masks in data_stream:

                # TODO REMOVE
                if l == 0:
                    break
                l -= 1

                batch_start_time = time.time()

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                # To handle torchvision.models
                if type(outputs) is OrderedDict:
                    outputs = outputs['out']

                loss = self.criterion(outputs, masks)

                preds = (outputs > self.threshold).long()

                # Update stats
                stats.update(epoch, 'loss', loss.item())
                stats.update(epoch, 'iou', iou(preds, masks).mean().item())
                stats.update(epoch, 'dice_coefficient', dice_coefficient(preds, masks).mean().item())
                stats.update(epoch, 'pixel_accuracy', pixel_accuracy(preds, masks).mean().item())
                stats.update(epoch, 'batch_eval_time', time.time() - batch_start_time)

        stats.update(epoch, 'epoch_eval_time', time.time() - epoch_start_time)
        if verbose:
            print(f"Evaluation ended in {stats.stats[epoch]['epoch_eval_time'][-1]} seconds")
            print('Average eval batch time', np.mean(stats.stats[epoch]['batch_eval_time']), 'seconds')
            print('Average loss', np.mean(stats.stats[epoch]['loss']))
            print('Metrics:')
            print('Average IoU \t\t\t', np.mean(stats.stats[epoch]['iou']))
            print('Average dice coefficient \t', np.mean(stats.stats[epoch]['dice_coefficient']))
            print('Average pixel accuracy \t\t', np.mean(stats.stats[epoch]['pixel_accuracy']))
            print(f"{'-' * 100}")

    def train(self,
              epochs: int,
              weights_dir: str = 'dmyhms',
              validate: bool = True,
              verbose: bool = False,
              limit: int = 2) -> Tuple[Statistics, Optional[Statistics]]:
        """
        Train the model

        :param epochs: number of epochs used to train
        :param weights_dir: path of the directory from the root used to save weights. If "dmyhms" uses the current date
        in DD_MM_YY_HH_MM_SS format
        :param validate: if True at the end of each epoch compute stats on the validation set
        :param verbose: if True print progress and info during training
        :param limit TODO remove it
        :return: statistics of training and if required of evaluation
        """

        if weights_dir is 'dmyhms':
            weights_dir = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

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

        # Check if the validation dataset has been defined
        if validate and self.validation_data_loader is None:
            validate = False

        if validate:
            eval_stats = Statistics(epochs,
                                    ['epoch_eval_time',
                                     'batch_eval_time',
                                     'loss',
                                     'iou',
                                     'dice_coefficient',
                                     'pixel_accuracy'],
                                    accumulate=True)
        else:
            eval_stats = None

        # Create directory containing weights
        if weights_dir is not None and weights_dir != '':
            weights_dir = os.path.join(self.root_path, weights_dir)
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            if verbose:
                print(f"Training epoch {epoch}/{epochs}:")
                data_stream = tqdm(iter(self.training_data_loader))
            else:
                data_stream = iter(self.training_data_loader)

            # TODO REMOVE
            l = limit
            for images, masks in data_stream:

                # TODO REMOVE
                if l == 0:
                    break
                l -= 1

                batch_start_time = time.time()

                images = images.to(self.device)
                masks = masks.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(images)

                # To handle torchvision.models
                if type(outputs) is OrderedDict:
                    outputs = outputs['out']

                loss = self.criterion(outputs, masks)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                preds = (outputs > self.threshold).long()

                # Update stats
                stats.update(epoch, 'loss', loss.item())
                stats.update(epoch, 'iou', iou(preds, masks).mean().item())
                stats.update(epoch, 'dice_coefficient', dice_coefficient(preds, masks).mean().item())
                stats.update(epoch, 'pixel_accuracy', pixel_accuracy(preds, masks).mean().item())
                stats.update(epoch, 'batch_train_time', time.time() - batch_start_time)

            if weights_dir is not None and weights_dir != '':
                torch.save(self.model.state_dict(), os.path.join(weights_dir, f'weights_{epoch}.pt'))

            stats.update(epoch, 'epoch_train_time', time.time() - epoch_start_time)
            if verbose:
                print(f"Epoch {epoch} ended in {stats.stats[epoch]['epoch_train_time'][-1]} seconds")
                print('Average training batch time', np.mean(stats.stats[epoch]['batch_train_time']), 'seconds')
                print('Average loss', np.mean(stats.stats[epoch]['loss']))
                print('Metrics:')
                print('Average IoU \t\t\t', np.mean(stats.stats[epoch]['iou']))
                print('Average dice coefficient \t', np.mean(stats.stats[epoch]['dice_coefficient']))
                print('Average pixel accuracy \t\t', np.mean(stats.stats[epoch]['pixel_accuracy']))

            if validate:
                if verbose:
                    print(f"{'-' * 100}\nValidation phase:")
                self.evaluate(epoch, eval_stats, verbose, limit=limit)

            if verbose:
                print(f"{'=' * 100}")

        return stats, eval_stats
