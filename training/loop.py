import os
from datetime import datetime

import numpy as np

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from preprocessing.dataset import HuBMAPDataset


class Statistics:
    def __init__(self, epochs, metrics, accumulate=False):
        """

        :param epochs: number of epochs
        :param metrics: list of metrics
        :param accumulate: True maintains lists, otherwise sum values at each update
        """
        self.accumulate = accumulate

        if accumulate:
            self.stats = {epoch: {s: [] for s in ['batches', 'loss', 'time'] + metrics}
                          for epoch in range(1, epochs + 1)}
        else:
            self.stats = {epoch: {s: (None, 0) for s in ['batches', 'loss', 'time'] + metrics}
                          for epoch in range(1, epochs + 1)}

    def update(self, epoch, stat, value):
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
            self.stats[epoch][stat][1] += 1

    def get_all_averages(self):
        """

        :return: averaged statistics at each epoch
        """
        if self.accumulate:
            return {e: {s: np.mean(self.stats[e][s]) for s in self.stats[e]} for e in self.stats}
        else:
            return {e: {s: (self.stats[e][s][0] / self.stats[e][s][1]) for s in self.stats[e]} for e in self.stats}

    def get_averaged_stat(self, stat):
        """

        :param stat: the selected state
        :return: the state stats averaged at each epoch
        """
        if self.accumulate:
            return [np.mean(self.stats[e][stat]) for e in self.stats]
        else:
            return [self.stats[e][stat][0] / self.stats[e][stat][1] for e in self.stats]


class Trainer:
    def __init__(self,
                 model: Module,
                 criterion: Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataset: HuBMAPDataset,
                 batch_size: int,
                 root_path: str,
                 device: str):
        """

        :param model: model to train
        :param criterion: loss function
        :param optimizer: optimizer used during training
        :param training_dataset: custom dataset to retrieve images and masks
        :param batch_size: size of bacthes used to create a DataLoader
        :param root_path: the path of the root
        :param device: device used
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        # TODO shuffle, workers etc..?
        self.training_data_loader = DataLoader(training_dataset, batch_size=batch_size)
        self.root_path = root_path
        self.device = device

    def train(self,
              epochs: int,
              weights_dir: str = datetime.now().strftime("%d_%m_%y_%H_%M_%S"),
              verbose: bool = False):
        """

        Train the model

        :param epochs: number of epochs used to train
        :param weights_dir: path of the directory from the root used to save weights
        :param verbose: if True print progress and info during training
        :return: statistics
        """

        # Set training mode
        self.model.train()

        # TODO Initialize statistics (timings, metrics, loss)

        # Create directory containing weights
        if weights_dir is not None and weights_dir != '':
            weights_dir = os.path.join(self.root_path, weights_dir)
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"Training epoch {epoch}:")

            for images, masks in iter(self.training_data_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs['out'], masks)

                y_pred = outputs['out'].data.cpu().numpy().ravel()
                y_true = masks.data.cpu().numpy().ravel()

                # TODO Metrics

                # backward and optimize
                loss.backward()
                self.optimizer.step()

                # TODO Update statistics

            if weights_dir is not None and weights_dir != '':
                self.model.save(os.path.join(weights_dir, f"weights_{epoch}.js"))
            
            if verbose:
                print(f"{'-'*100}")

    # TODO Return stats
