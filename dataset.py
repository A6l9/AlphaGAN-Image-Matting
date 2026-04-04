import csv
import random
import zipfile
from PIL import Image
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

import utils.train_utils as trn_utl
from cfg_loader import cfg
from transforms import TransformsPipeline


class CustomDataset(Dataset):
    def __init__(self, csv_path: Path, transforms: TransformsPipeline, train: bool = False) -> None:
        """
        Initialize the dataset from a CSV file and select the requested split.

        Args:
            csv_path (Path): Path to the CSV file containing dataset paths and split labels.
            transforms (TransformsPipeline): Transformation pipeline applied to loaded samples.
            train (bool): If True, use the 'train' split. Otherwise, use the 'test' split.
        """
        self.train = train
        split = "train" if self.train else "test"
        paths = pd.read_csv(csv_path)
        self.compos_paths = paths[paths["split"] == split].reset_index(drop=True)
        self.transforms = transforms

    def load_images(
        self,
        orig_path: Path,
        trim_path: Path,
        msk_path: Path,
    ) -> tuple[Image.Image, ...]:
        """
        Load the original image, trimap, and ground-truth mask from disk.

        Args:
            orig_path (Path): Path to the original RGB image.
            trim_path (Path): Path to the trimap image in grayscale mode.
            msk_path (Path): Path to the target mask image in grayscale mode.

        Returns:
            A tuple containing:
            - the original image in 'RGB' mode
            - the trimap in 'L' mode
            - the mask in 'L' mode
        """
        orig = Image.open(orig_path).convert("RGB")
        trim = Image.open(trim_path).convert("L")
        mask = Image.open(msk_path).convert("L")

        return orig, trim, mask

    def __len__(self) -> int:
        """
        Return the number of samples in the selected split.
        """
        return len(self.compos_paths)

    def __getitem__(self, index: int) -> dict:
        """
        Load one sample, apply transforms, and return the processed result.

        Args:
            index (int): Index of the sample in the selected split.

        Returns:
            A dictionary produced by the transformation pipeline.
        """
        row = self.compos_paths.iloc[index]

        orig = Path(row["original"])
        trim = Path(row["trimap"])
        mask = Path(row["mask"])

        orig, trim, mask = self.load_images(orig, trim, mask)

        result = self.transforms(orig, trim, mask, train=self.train, prepared=self.prepared_dataset)

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


def prepare_dataset_labels(
      dataset_path: Path,
      output_path: Path,
      test_ratio: float = 0.2,
  ) -> None:
      rows = []
      for part_dir in dataset_path.iterdir():
        for sample_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            compos_dir = sample_dir / "composite_crops"
            alpha_dir = sample_dir / "alpha_crops"
            trimap_dir = sample_dir / "trimap_crops"

            if not compos_dir.is_dir() or not alpha_dir.is_dir() or not trimap_dir.is_dir():
                continue

            compos_map = {p.name: p for p in compos_dir.glob("*.png")}
            alpha_map = {p.name: p for p in alpha_dir.glob("*.png")}
            trimap_map = {p.name: p for p in trimap_dir.glob("*.png")}

            common_names = sorted(compos_map.keys() & alpha_map.keys() & trimap_map.keys())

            for name in common_names:
                rows.append([
                    str(compos_map[name]),
                    str(trimap_map[name]),
                    str(alpha_map[name]),
                ])

      rng = random.Random(cfg.general.random_seed)
      rng.shuffle(rows)

      split_idx = int(len(rows) * (1 - test_ratio))
      train_rows = [row + ["train"] for row in rows[:split_idx]]
      test_rows = [row + ["test"] for row in rows[split_idx:]]

      with output_path.open("w", newline="", encoding="utf-8") as fp:
          writer = csv.writer(fp)
          writer.writerow(["original", "trimap", "mask", "split"])
          writer.writerows(train_rows + test_rows)


def unpack_archives(arch_path: Path, dst_path: Path) -> None:
    """Unpacks the archive by the 'arch_path' in the 'dst_path'

    Args:
        arch_path (Path): The archive path
        dst_path (Path): The destination path
    """
    with zipfile.ZipFile(arch_path) as zf:
        zf.extractall(dst_path)


if __name__ == "__main__":
    dt_path = Path(__file__).parent / "dataset" / "IMDatasetTiled.zip"
    dst_path = Path(__file__).parent / "dataset"

    unpack_archives(dt_path, dst_path)

    dt_path = Path(__file__).parent / "dataset" / "IMDatasetTiled"
    output_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"

    prepare_dataset_labels(dt_path, output_path)
