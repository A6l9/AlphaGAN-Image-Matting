from dataclasses import dataclass

import torch as tch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import losses as ls


@dataclass
class TrainComponents:
    device: tch.device
    epoch: int
    best_loss: float
    generator: nn.Module
    discriminator: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    g_optimizer: optim.Optimizer
    g_scheduler: lr.LRScheduler
    d_optimizer: optim.Optimizer
    d_scheduler: lr.LRScheduler
    l_alpha_loss: ls.BaseLoss
    l_comp_loss: ls.BaseLoss
    gan_loss: ls.BaseLoss
    writer: SummaryWriter


@dataclass
class BaseLossValues:
    def zeroing_loss_values(self) -> None:
        """
        Reset all loss values stored in the dataclass to zero.

        This method iterates over all instance attributes and sets
        their values to 0.0. It assumes that all fields represent
        numeric loss values.
        """
        for key in self.__dict__:
            self.__dict__[key] = 0.0


@dataclass 
class TrainLossValues(BaseLossValues):
    l1_alpha_loss: float = 0.0
    l1_compos_loss: float = 0.0
    bce_fake_d_loss: float = 0.0
    bce_real_d_loss: float = 0.0
    d_loss: float = 0.0
    g_loss: float = 0.0


@dataclass 
class TestLossValues(BaseLossValues):
    l1_alpha_loss: float = 0.0
    l1_compos_loss: float = 0.0
