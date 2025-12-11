from pathlib import Path

import torch as tch

from models import AlphaGenerator
import utils as utl
from dataset import CustomDataset
from cfg_loader import cfg
from transforms import TransformsPipeline
from schemas import TrainComponents


def train_one_epoch(epoch: int, components: TrainComponents) -> Dict[str, float]:
    """Run one training epoch for AlphaGAN (G + D)."""

    device        = components["device"]
    model         = components["model"]
    discriminator = components["discriminator"]
    train_loader  = components["train_loader"]
    g_optimizer   = components["g_optimizer"]
    g_scheduler   = components["g_scheduler"]
    d_optimizer   = components["d_optimizer"]
    d_scheduler   = components["d_scheduler"]
    l_alpha_loss  = components["l_alpha_loss"]
    l_comp_loss   = components["l_comp_loss"]
    gan_loss      = components["gan_loss"]
    writer        = components["writer"]

    # можно хранить в cfg
    lambda_alpha = getattr(cfg.loss, "lambda_alpha", 1.0)
    lambda_comp  = getattr(cfg.loss, "lambda_comp", 1.0)
    lambda_gan   = getattr(cfg.loss, "lambda_gan", 1.0)

    model.train()
    discriminator.train()

    running_D = 0.0
    running_G = 0.0
    running_alpha = 0.0
    running_comp = 0.0
    running_gan = 0.0

    global_step_start = epoch * len(train_loader)

    for i, batch in enumerate(train_loader):
        image    = batch["image"].to(device)
        trimap   = batch["trimap"].to(device)
        alpha_gt = batch["alpha_gt"].to(device)
        fg       = batch["fg"].to(device)
        bg       = batch["bg"].to(device)
        comp_gt  = batch["comp_gt"].to(device)

        # ========= 1) G: предсказание alpha =========
        # если у тебя генератор принимает только image — убери trimap
        alpha_pred = model(image, trimap)

        comp_pred = alpha_pred * fg + (1.0 - alpha_pred) * bg

        # ========= 2) обновляем дискриминатор =========
        d_optimizer.zero_grad()

        D_input_real = tch.cat([comp_gt, trimap], dim=1)          # (B, 4, H, W)
        D_input_fake = tch.cat([comp_pred.detach(), trimap], dim=1)

        D_real = discriminator(D_input_real)
        D_fake = discriminator(D_input_fake)

        loss_D_real = gan_loss[D_real, True]    # real → 1
        loss_D_fake = gan_loss[D_fake, False]   # fake → 0
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        loss_D.backward()
        d_optimizer.step()
        d_scheduler.step()

        # ========= 3) обновляем генератор =========
        g_optimizer.zero_grad()

        D_input_fake_for_G = tch.cat([comp_pred, trimap], dim=1)
        D_fake_for_G = discriminator(D_input_fake_for_G)

        loss_G_gan   = gan_loss[D_fake_for_G, True]               # хотим, чтобы fake считался real
        loss_G_alpha = l_alpha_loss[alpha_pred, alpha_gt]
        loss_G_comp  = l_comp_loss[alpha_pred, fg, bg, comp_gt]

        loss_G = (
            lambda_alpha * loss_G_alpha +
            lambda_comp  * loss_G_comp  +
            lambda_gan   * loss_G_gan
        )

        loss_G.backward()
        g_optimizer.step()
        g_scheduler.step()

        # ========= 4) статистика и логгинг =========
        running_D     += loss_D.item()
        running_G     += loss_G.item()
        running_alpha += loss_G_alpha.item()
        running_comp  += loss_G_comp.item()
        running_gan   += loss_G_gan.item()

        global_step = global_step_start + i

        if writer is not None:
            writer.add_scalar("train/loss_D",      loss_D.item(),      global_step)
            writer.add_scalar("train/loss_G",      loss_G.item(),      global_step)
            writer.add_scalar("train/loss_alpha",  loss_G_alpha.item(),global_step)
            writer.add_scalar("train/loss_comp",   loss_G_comp.item(), global_step)
            writer.add_scalar("train/loss_gan",    loss_G_gan.item(),  global_step)

    n = len(train_loader)
    return {
        "loss_D":      running_D / n,
        "loss_G":      running_G / n,
        "loss_alpha":  running_alpha / n,
        "loss_comp":   running_comp / n,
        "loss_gan":    running_gan / n,
    }


def train_pipeline(components: TrainComponents) -> None:
    """_summary_

    Args:
        components (dict): _description_
    """

    device = components["device"]
    model = components["model"]
    discriminator = components["discriminator"]
    train_loader = components["train_loader"]
    g_optimizer = components["g_optimizer"]
    g_scheduler = components["g_scheduler"]
    d_optimizer = components["d_optimizer"]
    d_scheduler = components["d_scheduler"]
    l_alpha_loss = components["l_alpha_loss"]
    l_comp_loss = components["l_comp_loss"]
    gan_loss = components["gan_loss"]
    writer = components["writer"]

    model.train()
    discriminator.train()

    running_D = 0.0
    running_G = 0.0
    running_alpha = 0.0
    running_comp = 0.0
    running_gan = 0.0