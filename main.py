from pathlib import Path

import torch as tch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models as mdl
import utils as utl
from dataset import CustomDataset
from cfg_loader import cfg
from transforms import TransformsPipeline
import losses as ls
from train import train_pipeline
from schemas import TrainComponents


def main(csv_path: Path) -> None:
    """Prepares components and setup the train loop

    Args:
        csv_path (Path): The path to the dataset labels
    """
    # Define an available device
    DEVICE = tch.device("cuda" if tch.cuda.is_available() else "cpu")
    
    # Define generator and dicsriminator
    generator = mdl.AlphaGenerator()
    generator.to(DEVICE)

    discriminator = mdl.PatchGANDiscriminator(4, nn.BatchNorm2d)
    discriminator.to(DEVICE)

    # Prepare train and test datasets
    dataset_train = CustomDataset(csv_path, train=True, transforms=TransformsPipeline)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=cfg.general.batch_size,
        shuffle=True,
        num_workers=utl.get_num_workers(),
        pin_memory=True
    )
    dataset_test = CustomDataset(csv_path, train=False, transforms=TransformsPipeline)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=cfg.general.batch_size,
        shuffle=False,
        num_workers=utl.get_num_workers(),
        pin_memory=True
    )

    # Define optimizer and scheduler for the discriminator and the generator
    g_optimizer = optim.AdamW(generator.parameters(),
                            lr=float(cfg.train.scheduler.start_lr),
                            weight_decay=float(cfg.train.optimizer.weight_decay)
                            )
    g_scheduler = CyclicLR(
                        g_optimizer,
                        base_lr=float(cfg.train.scheduler.start_lr),
                        max_lr=float(cfg.train.scheduler.end_lr), 
                        step_size_up=cfg.train.scheduler.step_size_up,
                        mode='triangular'
                    )
    
    d_optimizer = optim.AdamW(discriminator.parameters(),
                            lr=float(cfg.train.scheduler.start_lr),
                            weight_decay=float(cfg.train.optimizer.weight_decay)
                            )
    d_scheduler = CyclicLR(
                        d_optimizer,
                        base_lr=float(cfg.train.scheduler.start_lr),
                        max_lr=float(cfg.train.scheduler.end_lr), 
                        step_size_up=cfg.train.scheduler.step_size_up,
                        mode='triangular'
                    )
    
    # Define losses
    l_alpha_loss = ls.LAlphaLoss()
    l_comp_loss = ls.LCompositeLoss()
    gan_loss = ls.GANLoss()
        
    # Define the checkpoints dir and the logging dir paths
    checkpoints_dir = Path(cfg.general.checkpoints_dir).absolute()

    # Load the last checkpoint
    checkpoint = utl.load_checkpoint(checkpoints_dir, DEVICE)

    if checkpoint:
        print(utl.color("Checkpoint found, loading states...", "green"))

        generator.load_state_dict(checkpoint["model_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])

        g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state"])

        g_scheduler.load_state_dict(checkpoint["g_scheduler_state"])
        d_scheduler.load_state_dict(checkpoint["d_scheduler_state"])

        curr_epoch = checkpoint["epoch"] + 1
    else:
        print(utl.color("No checkpoint found, starting from scratch.", "green"))
        curr_epoch = 0

    # Define progress bar
    prog_bar = tqdm(iterable=enumerate(train_dataloader), unit="batch", desc="Training...", leave=True)

    # Create the tb logger
    with SummaryWriter(Path(cfg.train.logging.log_dir)) as writer:
        # Package it for train pipeline
        components = TrainComponents(
            device=DEVICE,
            generator=generator,
            epoch=curr_epoch,
            prog_bar=prog_bar,
            discriminator=discriminator,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            g_optimizer=g_optimizer,
            g_scheduler=g_scheduler,
            d_optimizer=d_optimizer,
            d_scheduler=d_scheduler,
            l_alpha_loss=l_alpha_loss,
            l_comp_loss=l_comp_loss,
            gan_loss=gan_loss,
            writer=writer
        )

        # Start the train pipeline
        train_pipeline(components)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"
    main(csv_path)
