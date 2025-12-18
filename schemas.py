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
class LossValues:
    l1_alpha_loss: float = 0.0
    l1_compos_loss: float = 0.0
    bce_fake_d_loss: float = 0.0
    bce_real_d_loss: float = 0.0
    d_loss: float = 0.0
    g_loss: float = 0.0
