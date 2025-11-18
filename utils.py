import random

import torch as tch

from loader import train_cfg as tcfg


def set_seed() -> None:
    random.seed(tcfg.train.random_seed)
    tch.random.manual_seed(tcfg.train.random_seed)
