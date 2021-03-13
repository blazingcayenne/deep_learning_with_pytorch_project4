import math

import numpy as np
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer

__all__ = ["OnceCycleLR"]


class OnceCycleLR(_LRScheduler):
    def __init__(self, optimizer, epochs, iters_per_epoch, min_lr_factor=0.05, max_lr=1.0):
        half_epochs = (epochs - 4) // 2
        half_iters = half_epochs * iters_per_epoch
        decay_epochs = epochs - 2 * half_epochs
        decay_iters = decay_epochs * iters_per_epoch

        lr_grow = np.linspace(min_lr_factor, max_lr, half_iters)
        lr_down = np.linspace(max_lr, min_lr_factor, half_iters)
        lr_decay = np.linspace(min_lr_factor, min_lr_factor * 0.01, 2 * decay_iters)
        self.learning_rates = np.concatenate((lr_grow, lr_down, lr_decay)) / max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.learning_rates[self.last_epoch] for base_lr in self.base_lrs]
