from pathlib import Path

import torch as tch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models as mdl
import utils.checkpointing as chkp_utl
import utils.terminal_utils as term_utl
import utils.train_utils as trn_utl
from dataset import CustomDataset
from cfg_loader import cfg
from transforms import TransformsPipeline
import losses as ls
from train_pipeline import train_pipeline
import schemas as sch


def get_discriminator(device: tch.device) -> sch.DComponents:
    """Build and configure the discriminator training components.

    This helper creates a discriminator, moves it to the target device,
    and initializes its optimizer, learning-rate scheduler, and GAN loss.

    Args:
        device (tch.device): Device to run the discriminator on (e.g. CPU or CUDA).

    Returns:
        sch.DComponents: A container with the discriminator module, optimizer,
            scheduler, and GAN loss instance.
    """
    # Define the discriminator
    discriminator = mdl.PatchGANDiscriminator(4, nn.BatchNorm2d, base_chann=12)
    discriminator.to(device)

    # Define optimizer and scheduler for the discriminator
    d_optimizer = optim.AdamW(discriminator.parameters(),
                            lr=float(cfg.train.D.scheduler.start_lr),
                            weight_decay=float(cfg.train.optimizer.weight_decay)
                            )
    d_scheduler = CyclicLR(
                        d_optimizer,
                        base_lr=float(cfg.train.D.scheduler.start_lr),
                        max_lr=float(cfg.train.D.scheduler.end_lr), 
                        step_size_up=cfg.train.D.scheduler.step_size_up,
                        mode='triangular'
                    )
    
    # Define the GAN loss
    gan_loss = ls.GANLoss()

    d_components = sch.DComponents(
        discriminator=discriminator,
        d_optimizer=d_optimizer,
        d_scheduler=d_scheduler,
        gan_loss=gan_loss
    )
    
    return d_components


def get_generator(device: tch.device) -> sch.GComponents:
    """Build and configure the generator training components.

    This helper creates a generator, moves it to the target device,
    and initializes its optimizer and learning-rate scheduler.

    Args:
        device (tch.device): Device to run the generator on (e.g. CPU or CUDA).

    Returns:
        sch.GComponents: A container with the generator module, optimizer
            and scheduler.
    """
    # Define generator
    generator = mdl.AlphaGenerator()
    generator.to(device)

    # Define optimizer and scheduler for the generator
    g_optimizer = optim.AdamW(generator.parameters(),
                            lr=float(cfg.train.G.scheduler.start_lr),
                            weight_decay=float(cfg.train.optimizer.weight_decay)
                            )
    g_scheduler = CyclicLR(
                        g_optimizer,
                        base_lr=float(cfg.train.G.scheduler.start_lr),
                        max_lr=float(cfg.train.G.scheduler.end_lr), 
                        step_size_up=cfg.train.G.scheduler.step_size_up,
                        mode='triangular'
                    )

    g_components = sch.GComponents(
        generator=generator,
        g_optimizer=g_optimizer,
        g_scheduler=g_scheduler
    )
    
    return g_components


def main(csv_path: Path) -> None:
    """Prepares components and setup the train loop
    
    Args:
        csv_path (Path): The path to the dataset labels
    """
    # Define an available device
    DEVICE = tch.device("cuda" if tch.cuda.is_available() else "cpu")

    transforms = TransformsPipeline()

    # Prepare train and test datasets
    dataset_train = CustomDataset(csv_path, train=True, transforms=transforms)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=cfg.general.batch_size,
        shuffle=True,
        num_workers=trn_utl.get_num_workers(),
        pin_memory=True
    )

    dataset_test = CustomDataset(csv_path, train=False, transforms=transforms)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=cfg.general.batch_size,
        shuffle=False,
        num_workers=trn_utl.get_num_workers(),
        pin_memory=True
    )
    
    # Define the AMP
    amp_status = bool(cfg.train.amp.use_amp)

    autocast = trn_utl.make_autocast(amp_status, DEVICE, cfg.train.amp.dtype)
    grad_scaler = trn_utl.make_grad_scaler(amp_status, DEVICE, cfg.train.amp.dtype, cfg.train.amp.use_grad_scaler)

    amp_components = sch.AMPComponents(
        autocast=autocast,
        grad_scaler=grad_scaler
    )
    
    # Define losses
    l_alpha_loss = ls.LAlphaLoss()
    l_comp_loss = ls.LCompositeLoss()

    # Define features extractor and perceptual loss
    features_extractor = mdl.VGG(layer_indices=(3, 8, 13, 15), model_type="vgg16")
    features_extractor.to(DEVICE)

    percept_loss = ls.PerceptualLoss(features_extractor)

    # Initialize the discriminator
    d_components = get_discriminator(DEVICE)

    # Initialize the generator
    g_components = get_generator(DEVICE)

    # Define the best test loss. Default 'inf'
    best_loss = float("inf")
        
    # Define the checkpoints dir and the logging dir paths
    checkpoints_dir = Path(cfg.general.checkpoints_dir).absolute()

    # Load the last checkpoint
    checkpoint = chkp_utl.load_checkpoint(checkpoints_dir, DEVICE)

    if checkpoint:
        print(term_utl.color("Checkpoint found, loading states...", "green"))

        g_components.generator.load_state_dict(checkpoint["model_state"])

        g_components.g_optimizer.load_state_dict(checkpoint["g_optimizer_state"])

        g_components.g_scheduler.load_state_dict(checkpoint["g_scheduler_state"])

        d_components.discriminator.load_state_dict(checkpoint["discriminator_state"])

        d_components.d_optimizer.load_state_dict(checkpoint["d_optimizer_state"])

        d_components.d_scheduler.load_state_dict(checkpoint["d_scheduler_state"])

        best_loss = checkpoint["best_loss"]

        curr_epoch = checkpoint["epoch"] + 1
    else:
        print(term_utl.color("No checkpoint found, starting from scratch.", "green"))
        curr_epoch = 1

    # Create the tb logger
    with SummaryWriter(Path(cfg.train.logging.log_dir)) as writer:
        # Package it for train pipeline
        components = sch.TrainComponents(
            device=DEVICE,
            epoch=curr_epoch,
            best_loss=best_loss,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            l_alpha_loss=l_alpha_loss,
            l_comp_loss=l_comp_loss,
            percept_loss=percept_loss,
            writer=writer,
            d_components=d_components,
            g_components=g_components,
            amp_components=amp_components
        )
                                           
        if not cfg.train.use_gan_loss:
            print(term_utl.color("Selected a training without a discriminator", "yellow"))

            components.use_gan_loss = bool(cfg.train.use_gan_loss)

        # Start the train pipeline
        train_pipeline(components)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "dataset" / "dataset_labels_short.csv"
    main(csv_path)
