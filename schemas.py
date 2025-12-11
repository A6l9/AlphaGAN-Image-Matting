import torch as tch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import TypedDict
import losses as ls


class TrainComponents(TypedDict):
    device: tch.device
    epoch: int
    prog_bar: tqdm
    model: nn.Module
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