import cv2
import albumentations as A


RANDOM_SEED = 1669

IMAGE_SIZE = (512, 512)

MASK_TRAIN_TFS = A.Compose(list([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.3,
        rotate_limit=40,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(0, 0, 0, 0),
        p=5.0
    ),
    A.pytorch.ToTensorV2()
]))

POST_COMP_TRAIN_TFS = A.Compose(list([
    A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.9, 1.1), p=0.8),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(std_range=(5.0, 25.0), p=0.4),
    A.ImageCompression(quality_range=(70, 100), p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.pytorch.ToTensorV2()
]))