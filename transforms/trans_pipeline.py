import numpy as np
import torch as tch
from PIL import Image
import albumentations as A

from cfg_loader import cfg
from . import models


class TransformsPipeline:
    compos_tfs = models.CompositingTransforms
    norm_tfs = models.NormalizeTransforms

    albu_train = A.Compose(
        [
            A.OneOf(
                [
                    A.ColorJitter(brightness=(0.4, 1.8), contrast=(0.3, 1.8), saturation=(0.3, 1.8), hue=(-0.1, 0.4), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.1, 0.2), p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomGamma(gamma_limit=(70, 140), p=1.0),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.1), per_channel=True, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(2, 10), p=1.0),
                    A.GaussianBlur(blur_limit=(2, 10), p=1.0),
                ],
                p=0.6,
            ),
            A.ImageCompression(quality_range=(75, 100), p=0.55),
        ],
        p=1.0,
    )

    albu_geom_train = A.Compose(
        [
            A.HorizontalFlip(p=0.6),
            A.Resize(height=cfg.train.resize_size, width=cfg.train.resize_size, p=1.0)
        ],
        additional_targets={
            "alpha": "image",
            "trimap": "mask",
        },
        p=1.0,
    )

    albu_geom_test = A.Compose(
        [
            A.Resize(height=cfg.test.resize_size, width=cfg.test.resize_size, p=1.0)
        ],
        additional_targets={
            "alpha": "image",
            "trimap": "mask",
        },
        p=1.0,
    )

    albu_norm = A.Compose(
        [
            A.ToTensorV2()
        ],
        additional_targets={
            "alpha": "image",
            "trimap": "mask",
        },
        p=1.0,
    )

    @classmethod
    def build_train_sample(cls, 
                            orig: np.ndarray,
                            trim: np.ndarray,
                            mask: np.ndarray
                            ) -> dict:
        """Build a training sample with geometric and image-level augmentations.

        Args:
            orig (np.ndarray): Input RGB image in HWC layout.
            trim (np.ndarray): Input trimap aligned with the input image.
            mask (np.ndarray): Input alpha mask aligned with the input image.

        Returns:
            dict: A dictionary with keys:
                - "compos": normalized tensor of shape (4, H, W) with RGB image and trimap.
                - "trim": normalized trimap tensor of shape (1, H, W).
                - "mask": normalized alpha mask tensor of shape (1, H, W).
                - "orig": normalized RGB tensor of shape (3, H, W).
        """
        geom_res = cls.albu_geom_train(image=orig, alpha=mask, trimap=trim)
        orig_aug = cls.albu_train(image=geom_res["image"])["image"]
        norm = cls.albu_norm(image=orig_aug, alpha=geom_res["alpha"], trimap=geom_res["trimap"])

        compos_trim = cls.compos_tfs.concat_image_and_trimap(norm["image"], norm["trimap"])
        compos_norm = cls.norm_tfs.normalize(compos_trim)
        mask_norm = cls.norm_tfs.normalize(norm["alpha"])
        orig_norm = cls.norm_tfs.normalize(norm["image"])

        return {
            "compos": compos_norm,
            "trim": compos_norm[3:],
            "mask": mask_norm,
            "orig": orig_norm
        }

    @classmethod
    def build_test_sample(cls, 
                        orig: np.ndarray,
                        trim: np.ndarray,
                        mask: np.ndarray,
                        ) -> dict:
        """Build a test sample with deterministic preprocessing only.

        Args:
            orig (np.ndarray): Input RGB image in HWC layout.
            trim (np.ndarray): Input trimap aligned with the input image.
            mask (np.ndarray): Input alpha mask aligned with the input image.

        Returns:
            dict: A dictionary with keys:
                - "compos": normalized tensor of shape (4, H, W) with RGB image and trimap.
                - "trim": normalized trimap tensor of shape (1, H, W).
                - "mask": normalized alpha mask tensor of shape (1, H, W).
                - "orig": normalized RGB tensor of shape (3, H, W).
        """
        geom_res = cls.albu_geom_test(image=orig, alpha=mask, trimap=trim)
        norm = cls.albu_norm(image=geom_res["image"], alpha=geom_res["alpha"], trimap=geom_res["trimap"])

        compos_trim = cls.compos_tfs.concat_image_and_trimap(norm["image"], norm["trimap"])
        compos_norm = cls.norm_tfs.normalize(compos_trim)
        mask_norm = cls.norm_tfs.normalize(norm["alpha"])
        orig_norm = cls.norm_tfs.normalize(norm["image"])

        return {
            "compos": compos_norm,
            "trim": compos_norm[3:],
            "mask": mask_norm,
            "orig": orig_norm
        }
    
    @classmethod
    def __call__(cls, 
                 orig: Image.Image, 
                 trim: Image.Image, 
                 mask: Image.Image,
                 train: bool=True) -> dict:
        """Run the full preprocessing pipeline for a single sample.

        Args:
            orig (Image.Image): Input RGB PIL image.
            trim (Image.Image): Input trimap PIL image in grayscale mode.
            mask (Image.Image): Input alpha mask PIL image in grayscale mode.
            train (bool, optional): If True, apply training augmentations.
                Otherwise, use deterministic test preprocessing. Defaults to True.

        Returns:
            dict: A dictionary with keys:
                - "compos": normalized tensor of shape (4, H, W) with RGB image and trimap.
                - "trim": normalized trimap tensor of shape (1, H, W).
                - "mask": normalized alpha mask tensor of shape (1, H, W).
                - "orig": normalized RGB tensor of shape (3, H, W).
        """
        orig_np = cls.norm_tfs.to_numpy(orig)
        trim_np = cls.norm_tfs.to_numpy(trim)
        mask_np = cls.norm_tfs.to_numpy(mask)

        if train:
            return cls.build_train_sample(orig_np, trim_np, mask_np)

        return cls.build_test_sample(orig_np, trim_np, mask_np)
