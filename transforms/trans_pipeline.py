import torch as tch
from PIL import Image
import torchvision.transforms.v2 as T

from cfg_loader import cfg
from . import models
import utils as utl


class TransformsPipeline:
    geom_tfs = models.GeometryTransforms
    compos_tfs = models.CompositingTransforms
    crop_tfs = models.CropTransforms
    norm_tfs = models.NormalizeTransforms

    @classmethod
    def build_train_sample(cls, orig: tch.Tensor, 
                           trim: tch.Tensor, 
                           mask: tch.Tensor, 
                           bg: tch.Tensor
                           ) -> dict:
        """Builds a train sample and applies augmentations

        Args:
            orig (tch.Tensor): Foreground image as a tensor
            mask (tch.Tensor): Mask image as a tensor
            trim (tch.Tensor): Trimap image as a tensor
            bg (tch.Tensor): Background image as a tensor

        Returns:
            dict: The prepared composite, trimap, mask, foreground image and background
        """
        orig_rot, trim_rot, mask_rot = cls.geom_tfs.random_rotate(orig, trim, mask, prob=0.6)
        orig_scl, trim_scl, mask_scl = cls.geom_tfs.random_scale(orig_rot, trim_rot, mask_rot, prob=0.6)
        orig_fl, trim_fl, mask_fl = cls.geom_tfs.random_hflip(orig_scl, trim_scl, mask_scl, prob=0.6)

        bg_res = cls.geom_tfs.random_background_scale(bg)

        orig_res, trim_res, mask_res = cls.geom_tfs.resize_to_fit_background(orig_fl, trim_fl, mask_fl, bg_res)
        compos, trim_comp, mask_comp, orig_comp, bg_comp = cls.compos_tfs.random_placement(orig_res, trim_res, mask_res, bg_res)

        compos_crop, trim_crop, mask_crop, orig_crop, bg_crop = cls.crop_tfs.random_unknown_crop(compos, 
                                                                                            trim_comp, 
                                                                                            mask_comp,
                                                                                            orig_comp,
                                                                                            bg_comp,
                                                                                            cfg.train.crop_size)

        compos_trim = cls.compos_tfs.concat_image_and_trimap(compos_crop, trim_crop)
        compos_norm = cls.norm_tfs.normalize(compos_trim, imgnet=True)
        mask_norm = cls.norm_tfs.normalize(mask_crop)

        orig_norm = cls.norm_tfs.normalize(orig_crop, imgnet=True)
        bg_norm = cls.norm_tfs.normalize(bg_crop, imgnet=True)

        return {
            "compos": compos_norm,
            "trim": compos_norm[3:],
            "mask": mask_norm,
            "orig": orig_norm,
            "bg": bg_norm
        }

    @classmethod
    def build_test_sample(cls, orig: tch.Tensor, 
                          trim: tch.Tensor, 
                          mask: tch.Tensor, 
                          bg: tch.Tensor
                          ) -> dict:
        """Builds a test sample without any augmentations

        Args:
            orig (tch.Tensor): Foreground image as a tensor
            mask (tch.Tensor): Mask image as a tensor
            trim (tch.Tensor): Trimap image as a tensor
            bg (tch.Tensor): Background image as a tensor

        Returns:
            dict: The prepared orig, trimap and mask
        """
        orig_res, trim_res, mask_res = cls.geom_tfs.resize_to_fit_background(orig, trim, mask, bg)
        compos, trim_comp, mask_comp, orig_comp, bg_comp = cls.compos_tfs.center_placement(orig_res, trim_res, mask_res, bg)

        with utl.set_seed(cfg.general.random_seed, use_cuda=False):
            compos_crop, trim_crop, mask_crop, orig_crop, bg_crop = cls.crop_tfs.random_unknown_crop(compos, 
                                                                                                trim_comp, 
                                                                                                mask_comp,
                                                                                                orig_comp,
                                                                                                bg_comp,
                                                                                                cfg.test.crop_size
                                                                                                )

        compos_trim = cls.compos_tfs.concat_image_and_trimap(compos_crop, trim_crop)
        compos_norm = cls.norm_tfs.normalize(compos_trim, imgnet=True)
        mask_norm = cls.norm_tfs.normalize(mask_crop)
        
        orig_norm = cls.norm_tfs.normalize(orig_crop, imgnet=True)
        bg_norm = cls.norm_tfs.normalize(bg_crop, imgnet=True)

        return {
            "compos": compos_norm,
            "trim": compos_norm[3:],
            "mask": mask_norm,
            "orig": orig_norm,
            "bg": bg_norm
        }
    
    @classmethod
    def __call__(cls, orig: Image.Image, 
                 trim: Image.Image, 
                 mask: Image.Image, 
                 bg: Image.Image,
                 train: bool=True) -> dict:
        """Full augmentation and compositing pipeline:

        - converts all PIL images to tensors;
        - applies random rotation, scale, and horizontal flip to the foreground
          and corresponding trimap/mask;
        - resizes the foreground to fit inside the background;
        - composites foreground onto a random location on the background;
        - performs a gray-region crop around unknown trimap area;
        - normalizes the composite and scales the mask to [0, 1];
        - concatenates trimap as an additional input channel.

        Args:
            orig (Image.Image): Foreground RGB PIL image.
            trim (Image.Image): Trimap PIL image (mode "L").
            mask (Image.Image): Mask PIL image (mode "L").
            bg (Image.Image): Background RGB PIL image.
            train (bool, optional): If True.

        Returns:
            dict: A dictionary with keys:
                - "compos": tensor of shape (4, H, W) (RGB + trimap).
                - "trim": full-size trimap tensor of shape (1, H, W).
                - "mask": full-size mask tensor of shape (1, H, W) in [0, 1].
        """
        orig_ten = cls.norm_tfs.to_tensor(orig)
        trim_ten = cls.norm_tfs.to_tensor(trim)
        mask_ten = cls.norm_tfs.to_tensor(mask) 
        bg_ten = cls.norm_tfs.to_tensor(bg)

        if train:
            return cls.build_train_sample(orig_ten, trim_ten, mask_ten, bg_ten)

        return cls.build_test_sample(orig_ten, trim_ten, mask_ten, bg_ten)
