import enum
import math
import warnings
import os
import time
import pickle
from datetime import datetime
from collections import OrderedDict
from typing import Union, Optional, List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchvision.transforms import transforms
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from evaluation.metrics import iou, pixel_accuracy, dice_coefficient
from preprocessing.dataset import HuBMAPDataset, denormalize_images
from visualization.visualize_data import display_images_and_masks


class TrainerVerbosity(enum.Enum):
    """
    Enumerator used to specify the verbosity of the training loop
    """
    # Print only information regarding the progress
    PROGRESS = 1
    # Print saved statistics at each epoch
    STATISTICS = 2
    # Print involved tensors
    TENSORS = 3
    # Show involved images
    IMAGES = 4
    # Frequency for tensors and images during training and evaluation
    TENSORS_TRAIN_FREQ = 53
    TENSORS_EVAL_FREQ = 30


class Statistics:
    """
    Class used to register and keep track of the training and evaluating statistics. It creates a dictionary mapping
    each epoch and each stat with a list of values or an accumulated value.
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
        :return: averaged statistics at each epoch (including those not already computed)
        """

        if self.accumulate:
            return {e: {s: np.mean(self.stats[e][s]) for s in self.stats[e]} for e in self.stats}
        else:
            return {e: {s: (self.stats[e][s][0] / self.stats[e][s][1]) for s in self.stats[e]} for e in self.stats}

    def get_averaged_stat(self, stat: str) -> List:
        """
        :param stat: the selected state
        :return: the state stats averaged at each epoch (including those not already computed)
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


class SchedulerWrapper:
    """
    Wrap a PyTorch scheduler providing utilities to reset its state, e.g. to be performed at the end of an epoch.
    """

    def __init__(self,
                 scheduler: object,
                 reset_strategy: Optional[Callable[[int], bool]]):
        """
        :param scheduler: a learning rate scheduler
        :param reset_strategy: a Callable that return True when reset must be performed based on the passed epoch
        """

        self.scheduler = scheduler
        self.reset_strategy = reset_strategy

        # Save scheduler and optimizer's initial state to recover later if necessary
        self.scheduler_state_dict = scheduler.state_dict()
        self.optimizer_state_dict = self.scheduler.optimizer.state_dict()

    def step(self, params: Dict = None):
        """
        The classical scheduler's step
        """

        if params is None:
            params = {}
        self.scheduler.step(**params)

    def reset(self, epoch):
        """
        Reset the scheduler and related optimizer to its initial state
        :param epoch: current epoch
        """

        if self.reset_strategy and self.reset_strategy(epoch):
            self.scheduler.load_state_dict(self.scheduler_state_dict)
            self.scheduler.optimizer.load_state_dict(self.optimizer_state_dict)


class EarlyStopping:
    """
    Class responsible for performing early stopping based on the observation of no improvement for more than a certain
    number of epochs.
    """

    def __init__(self,
                 num_epochs_to_stop: int,
                 delta: float = 1e-5):
        """
        :param num_epochs_to_stop: maximum number of epochs to stop when validation loss is not improving
        :param delta: minimum difference between best validation loss and the current one to consider the new an
        improvement
        """

        self.num_epochs_to_stop = num_epochs_to_stop
        self.delta = delta
        self.min_validation_loss = None
        self.epochs_no_improve = 0

    def step(self, val_loss: float) -> bool:
        """
        Check whether the training should stop or not and update internal records.

        :param val_loss: current epoch validation loss (usually an average of an epoch)
        :return: True if the training should be stopped
        """

        if self.min_validation_loss is None:
            self.min_validation_loss = val_loss
            return False

        # Update current best validation loss and reset counter
        if val_loss + self.delta < self.min_validation_loss:
            self.epochs_no_improve = 0
            self.min_validation_loss = val_loss
        # Update counter and check whether maximum number of epochs of no improvement has been reached
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.num_epochs_to_stop:
                return True

        return False


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
        # Get mean and std from datasets to denormalize and show images
        self.mean = training_dataset.mean
        self.std = training_dataset.std
        self.root_path = root_path
        self.device = device

    def print_stats(self,
                    stats: Statistics,
                    epoch: int,
                    print_epoch_time: bool):
        """
        Print current epoch statistics

        :param stats: statistics
        :param epoch: current epoch
        :param print_epoch_time: if True prints the epoch/evaluation duration
        """

        # Time is already indicated by tqdm
        if print_epoch_time:
            print('Epoch/Evaluation ended in', round(stats.stats[epoch]['epoch_time'][-1]), 'seconds')
        print('Average batch time \t', round(np.mean(stats.stats[epoch]['batch_time'])), 'seconds')
        print('Average loss \t\t\t', round(np.mean(stats.stats[epoch]['loss']), 4))
        print('Metrics:')
        print('Average IoU \t\t\t', np.mean(stats.stats[epoch]['iou']))
        print('Average dice coefficient \t', np.mean(stats.stats[epoch]['dice_coefficient']))
        print('Average pixel accuracy \t\t', np.mean(stats.stats[epoch]['pixel_accuracy']))

    def evaluate(self,
                 epoch: int,
                 stats: Statistics,
                 verbosity_level: List[TrainerVerbosity] = (),
                 limit: int = math.inf) -> None:
        """
        Method used to evaluate the model

        :param epoch: current epoch
        :param stats: statistics tracker
        :param verbosity_level: List containing different keys for each type of requested information
        :param limit TODO remove this
        """

        with torch.no_grad():
            epoch_start_time = time.time()

            if TrainerVerbosity.PROGRESS in verbosity_level:
                data_stream = tqdm(iter(self.validation_data_loader))
            else:
                data_stream = iter(self.validation_data_loader)

            # TODO remove this
            l = 0
            for images, masks in data_stream:

                # TODO remove this
                if l == limit:
                    break
                l += 1

                batch_start_time = time.time()

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                # To handle torchvision.models
                if type(outputs) is OrderedDict:
                    outputs = outputs['out']

                loss = self.criterion(outputs, masks)

                preds = (outputs > self.threshold).long()

                # Print tensors
                if TrainerVerbosity.TENSORS in verbosity_level and l % TrainerVerbosity.TENSORS_EVAL_FREQ.value == 0:
                    print(f'Image ({images.shape}):\n{images}\n')
                    print(f'Max and min value: {images.max().item()}, {images.min().item()}\n')
                    print(f'Masks ({masks.shape}):\n{masks}\n')
                    print(f'Outputs ({outputs.shape}):\n{outputs}\n')
                    print(f'Max and min value: {outputs.max().item()}, {outputs.min().item()}\n')
                    print(f'Predictions ({preds.shape}):\n{preds}\n')
                    print(f'Loss:\n{loss.item()}\n')

                # Show images
                if TrainerVerbosity.IMAGES in verbosity_level and l % TrainerVerbosity.TENSORS_EVAL_FREQ.value == 0:
                    masks = masks.cpu()
                    outputs = outputs.cpu()
                    preds = preds.cpu()
                    images = images.detach().cpu()
                    for i in range(len(images)):
                        # Denormalize and then transform to PIL image
                        denormalized_images = transforms.ToPILImage()(denormalize_images(images[i],
                                                                                         self.mean,
                                                                                         self.std))
                        display_images_and_masks(images=[denormalized_images] * 3,
                                                 masks=[masks[i].detach(),
                                                        outputs[i].detach().squeeze(0),
                                                        preds[i].detach().squeeze(0)])

                # Update stats
                stats.update(epoch, 'loss', loss.item())
                stats.update(epoch, 'iou', iou(preds, masks).mean().item())
                stats.update(epoch, 'dice_coefficient', dice_coefficient(preds, masks).mean().item())
                stats.update(epoch, 'pixel_accuracy', pixel_accuracy(preds, masks).mean().item())
                stats.update(epoch, 'batch_time', time.time() - batch_start_time)

        stats.update(epoch, 'epoch_time', time.time() - epoch_start_time)
        # Print stats
        if TrainerVerbosity.STATISTICS in verbosity_level:
            self.print_stats(stats, epoch, TrainerVerbosity.PROGRESS not in verbosity_level)
            print(f"{'-' * 100}")

    def train(self,
              epochs: int,
              saving_frequency: int = 1,
              scheduler: Optional[SchedulerWrapper] = None,
              weights_dir: str = 'dmyhms',
              evaluate: bool = True,
              early_stopping: Optional[EarlyStopping] = None,
              verbosity_level: List[TrainerVerbosity] = (),
              evaluation_verbosity_level: List[TrainerVerbosity] = (),
              limit: int = math.inf,
              evaluation_limit: int = math.inf) -> Tuple[Statistics, Optional[Statistics]]:
        """
        Train the model

        :param epochs: number of epochs used to train
        :param weights_dir: path of the directory from the root used to save weights. If "dmyhms" uses the current date
        in DD_MM_YY_HH_MM_SS format
        :param saving_frequency: indicates the epoch rate of weights saving
        :param scheduler: a learning rate scheduler
        :param evaluate: if True at the end of each epoch compute stats on the validation set
        :param early_stopping: early stopping policy and tracker
        :param verbosity_level: list containing different keys for each type of requested information (training)
        :param evaluation_verbosity_level: list containing different keys for each type of requested information
        :param limit: the number of batches to debug TODO remove this
        :param evaluation_limit: the number of batches to debug in evaluation TODO remove this
        :return: statistics of training and if required of evaluation
        """

        if weights_dir is 'dmyhms':
            weights_dir = datetime.now().strftime('%d_%m_%y_%H_%M_%S')

        # Set training mode
        self.model.train()

        stats_keys = ['epoch_time',
                      'batch_time',
                      'loss',
                      'iou',
                      'dice_coefficient',
                      'pixel_accuracy']

        # Initialize statistics
        stats = Statistics(epochs,
                           stats_keys,
                           accumulate=True)

        # Check if the validation dataset has been defined
        if evaluate and self.validation_data_loader is None:
            evaluate = False
            warnings.warn('evaluate is True but no validation dataset has been passed! Evaluation will be skipped.')

        if not evaluate and early_stopping is not None:
            raise Exception('Cannot perform early stopping without validation phase!')

        eval_stats = None
        if evaluate:
            eval_stats = Statistics(epochs,
                                    stats_keys,
                                    accumulate=True)

        # Create directory containing weights
        if weights_dir is not None and weights_dir != '':
            weights_dir = os.path.join(self.root_path, weights_dir)
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            if TrainerVerbosity.PROGRESS in verbosity_level:
                print(f'Training epoch {epoch}/{epochs}:')
                data_stream = tqdm(iter(self.training_data_loader))
            else:
                data_stream = iter(self.training_data_loader)

            i = 0
            for images, masks in data_stream:

                # TODO remove this
                if i == limit:
                    break
                i += 1

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

                # If the scheduler is defined update it
                if scheduler:
                    if type(scheduler.scheduler) is CosineAnnealingWarmRestarts:
                        scheduler.step({'epoch': epoch + i / len(self.training_data_loader)})
                    elif type(scheduler.scheduler) is not ReduceLROnPlateau:
                        scheduler.step()

                preds = (outputs > self.threshold).long()

                # Print tensors
                if TrainerVerbosity.TENSORS in verbosity_level and i % TrainerVerbosity.TENSORS_TRAIN_FREQ.value == 0:
                    print(f'Image ({images.shape}):\n{images}\n')
                    print(f'Max and min value: {images.max().item()}, {images.min().item()}\n')
                    print(f'Masks ({masks.shape}):\n{masks}\n')
                    print(f'Outputs ({outputs.shape}):\n{outputs}\n')
                    print(f'Max and min value: {outputs.max().item()}, {outputs.min().item()}\n')
                    print(f'Predictions ({preds.shape}):\n{preds}\n')
                    print(f'Loss:\n{loss.item()}\n')

                # Show images
                if TrainerVerbosity.IMAGES in verbosity_level and i % TrainerVerbosity.TENSORS_TRAIN_FREQ.value == 0:
                    masks = masks.cpu()
                    outputs = outputs.cpu()
                    preds = preds.cpu()
                    images = images.detach().cpu()
                    for i in range(len(images)):
                        # Denormalize and then transform to PIL image
                        denormalized_images = transforms.ToPILImage()(denormalize_images(images[i],
                                                                                         self.mean,
                                                                                         self.std))
                        display_images_and_masks(images=[denormalized_images] * 3,
                                                 masks=[masks[i].detach(),
                                                        outputs[i].detach().squeeze(0),
                                                        preds[i].detach().squeeze(0)])

                # Update stats
                stats.update(epoch, 'loss', loss.item())
                stats.update(epoch, 'iou', iou(preds, masks).mean().item())
                stats.update(epoch, 'dice_coefficient', dice_coefficient(preds, masks).mean().item())
                stats.update(epoch, 'pixel_accuracy', pixel_accuracy(preds, masks).mean().item())
                stats.update(epoch, 'batch_time', time.time() - batch_start_time)

            if weights_dir is not None and weights_dir != '' and (epoch - 1) % saving_frequency == 0:
                torch.save(self.model.state_dict(), os.path.join(weights_dir, f'weights_{epoch}.pt'))

            stats.update(epoch, 'epoch_time', time.time() - epoch_start_time)

            # Print stats
            if TrainerVerbosity.STATISTICS in verbosity_level:
                self.print_stats(stats, epoch, TrainerVerbosity.PROGRESS not in verbosity_level)

            if evaluate:
                if TrainerVerbosity.PROGRESS in verbosity_level:
                    print(f"{'-' * 100}\nValidation phase:")
                self.evaluate(epoch, eval_stats, evaluation_verbosity_level, limit=evaluation_limit)

            if scheduler:
                if type(scheduler.scheduler) is ReduceLROnPlateau:
                    scheduler.step({'metrics': eval_stats.get_averaged_stat('loss')[epoch - 1]})

                # Apply scheduler resetting
                scheduler.reset(epoch)

            if early_stopping:
                if early_stopping.step(eval_stats.get_averaged_stat('loss')[epoch - 1]):
                    if TrainerVerbosity.PROGRESS in verbosity_level:
                        print('Early Stopping!')
                    return stats, eval_stats

            if TrainerVerbosity.PROGRESS in verbosity_level:
                print(f"{'=' * 100}")

        return stats, eval_stats
