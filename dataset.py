import csv
import random
from PIL import Image
from pathlib import Path

import pandas as pd
from box import Box
from torch.utils.data import Dataset

import utils as utl
from cfg_loader import cfg
from transforms import TransformsPipeline


class CustomDataset(Dataset):
    def __init__(self, csv_path: Path, config: Box, transforms: TransformsPipeline, train: bool=False) -> None:
        self.train = train
        split = "train" if self.train else "test"
        paths = pd.read_csv(csv_path)
        self.compos_paths = paths[paths["split"] == split].reset_index(drop=True)
        self.config = config
        self.transforms = transforms
    
    def load_images(self, 
                  orig_path: Path, 
                  trim_path: Path,
                  msk_path: Path, 
                  bg_path: Path) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        """
        Loads the foreground (orig), trimap, mask, and background images.

        Args:
            orig_path (Path): Path to the original RGB foreground image.
            trim_path (Path): Path to the trimap (L mode).
            msk_path (Path): Path to the mask (L mode).
            bg_path (Path): Path to the background RGB image.

        Returns:
            tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
                (orig, trim, mask, bg) PIL images.
        """
        orig = Image.open(orig_path).convert("RGB")
        trim = Image.open(trim_path).convert("L")
        mask = Image.open(msk_path).convert("L")
        bg = Image.open(bg_path).convert("RGB")

        return orig, trim, mask, bg

    def __len__(self) -> int:
        return len(self.compos_paths)
    
    def __getitem__(self, index: int) -> dict:
        """Takes an index of image paths and then build test/train sample. 

        Args:
            index (int): The index of image paths from a csv file.

        Returns:
            dict: The composite and the mask as tensors into a dictionary
        """
        row = self.compos_paths.iloc[index]
        
        orig = Path(row["original"])
        trim = Path(row["trimap"])
        mask = Path(row["mask"])
        bg = Path(row["background"])

        orig, trim, mask, bg = self.load_images(orig, trim, mask, bg)

        result = self.transforms(orig, trim, mask, bg, train=self.train)

        return result


def check_dirs(path: Path, required: list[str]) -> bool:
    """Checks required dirs

    Args:
        path (Path): The path to the directory
        required (list[str]): The list of the names required directories

    Returns:
        bool: The check status
    """
    dirs_from_path = {p.name for p in path.iterdir() if p.is_dir()}

    return set(required) <= dirs_from_path


def unpack_dir(path: Path, recur: bool=False) -> dict:
    """Unpack directory content to the dictionary

    Args:
        path (Path): The path to the directory
        recur (bool, optional): If the directory has nested directories. Defaults to False.

    Returns:
        dict: The result dictionary
    """
    if not recur:
        return  {elem.stem: elem for elem in path.iterdir()}
    else:
        images = {}
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            for elem in path.rglob(pattern):
                images[elem.stem] = elem
        
        return images


def prepare_labels(fg_path: Path,
                   bg_path: Path,
                   output_path: Path,
                   required_fg: list[str] | None=None,
                   test_ratio: float=0.2
                   ) -> None:
    """Create random composite pairs from foregrounds and backgrounds and save them to a CSV file.

    Args:
        fg_path (Path): The path to the foreground images, their masks and trimaps
        bg_path (Path): The path to the background images
        output_path (Path): The output path to the csv file
        required_fg (list): The list of the names required directories
    """
    with utl.set_seed(cfg.general.random_seed):
        if required_fg is None:
            required_fg = ["mask", "trimap", "original"]
            
        if not check_dirs(fg_path, required_fg):
            raise FileNotFoundError(f"The directories [{required_fg}] are required")
        
        fg_masks = unpack_dir(fg_path / required_fg[0])
        fg_trimaps = unpack_dir(fg_path / required_fg[1])
        fg_origs = unpack_dir(fg_path / required_fg[2])

        backgrounds = unpack_dir(bg_path, recur=True)
        
        rows = []

        for bg in backgrounds:
            orig = random.choice(list(fg_origs.keys()))
            if fg_masks.get(orig) and fg_trimaps.get(orig):
                orig_pth = fg_origs.get(orig)
                trim_pth = fg_trimaps.get(orig)
                msk_pth = fg_masks.get(orig)
                bg_pth = backgrounds.get(bg)

                rows.append([str(orig_pth), str(trim_pth), str(msk_pth), str(bg_pth)])

        random.shuffle(rows)
        split_idx = int(len(rows) * (1 - test_ratio))
        train_set = list(map(lambda x: x + ["train"], rows[:split_idx]))
        test_set = list(map(lambda x: x + ["test"], rows[split_idx:]))
        train_set.extend(test_set)

        with output_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["original", "trimap", "mask", "background", "split"])
            writer.writerows(train_set)


if __name__ == "__main__":
    fg_path = Path(__file__).parent / "dataset" / "AIM-500"
    bg_path = Path(__file__).parent / "dataset" / "BG20K"
    output_path = Path(__file__).parent / "dataset_labels.csv"

    prepare_labels(fg_path, bg_path, output_path)
