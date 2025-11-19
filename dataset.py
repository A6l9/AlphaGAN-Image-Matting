import csv
import random
from PIL import Image
from pathlib import Path

import torch as tch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as F

import utils as utl
from transforms import CustomTransforms


class CustomDataset(Dataset):
    def __init__(self, csv_path: Path, train: bool=False) -> None:
        split = "train" if train else "test"
        paths = pd.read_csv(csv_path)
        self.compos_paths = paths[paths["split"] == split].reset_index(drop=True)
        self.transforms = CustomTransforms

    def composite(self, 
                  orig_path: Path, 
                  trim_path: Path,
                  msk_path: Path, 
                  bg_path: Path) -> tuple[tch.Tensor, tch.Tensor]:
        """Creates a composite image by cutting out the foreground using the mask,
        placing it on a random position on the background, and attaching the
        trimap as an additional (4th) channel.

        Steps performed:
        - Load original RGBA image, mask, trimap, and background
        - Resize the original so it fully fits inside the background
        - Cut out the foreground using the mask as alpha
        - Paste the cut-out foreground onto a random location on the background
        - Append the trimap as a 4th channel (RGB + Trimap)

        Args:
            orig_path (Path): Path to the original image (RGBA expected)
            trim_path (Path): Path to the trimap image (L mode)
            msk_path  (Path): Path to the mask image (L mode)
            bg_path   (Path): Path to the background image (RGBA expected)

        Returns:
            tuple[tch.Tensor, tch.Tensor]: 
                - Composite image in (H, W, 4) format, float32 or uint8
                    depending on the input images.
                - The mask of the foreground image.
            
        """
        orig = Image.open(orig_path).convert("RGBA")
        trim = Image.open(trim_path).convert("L")
        mask = Image.open(msk_path).convert("L")
        bg = Image.open(bg_path).convert("RGBA")

        # Apply the transformations to foreground image, trimap and mask
        orig_rot, trim_rot, mask_rot = self.transforms.random_rotate(orig, trim, mask, prob=0.6)
        orig_hfl, trim_hfl, mask_hfl = self.transforms.random_hflip(orig_rot, trim_rot, mask_rot, 0.7)
        orig_scl, trim_scl, mask_scl = self.transforms.random_scale(orig_hfl, trim_hfl, mask_hfl, prob=0.6)

        w_bg, h_bg = bg.size
        w_orig, h_orig = orig_scl.size

        scale_rate = min(1.0, w_bg / w_orig, h_bg / h_orig)
        if scale_rate < 1.0:
            # If the scale rate < 1.0, so we need to resize the orig, trimap and mask to fit it in the background 
            new_h = int(h_orig * scale_rate)
            new_w = int(w_orig * scale_rate)
            orig_scl = orig_scl.resize((new_w, new_h), Image.Resampling.LANCZOS)
            trim_scl = trim_scl.resize((new_w, new_h), Image.Resampling.NEAREST)
            mask_scl = mask_scl.resize((new_w, new_h), Image.Resampling.NEAREST)
            w_orig, h_orig = new_w, new_h
        
        x_rand = random.randint(0, w_bg - w_orig)
        y_rand = random.randint(0, h_bg - h_orig)

        cut = Image.new("RGBA", size=(w_orig, h_orig))
        cut.paste(orig_scl, (0, 0), mask_scl)
        bg.paste(cut, (x_rand, y_rand), mask_scl)
        
        # Paste the trimap and mask to the background template
        trim_big = Image.new("L", bg.size, color=0)
        trim_big.paste(trim_scl, (x_rand, y_rand))

        mask_big = Image.new("L", bg.size, color=0)
        mask_big.paste(mask_scl, (x_rand, y_rand))
        
        # Convert it to tensors
        bg_t, trim_t, mask_t = F.pil_to_tensor(bg), F.pil_to_tensor(trim_big), F.pil_to_tensor(mask_big)

        comp_crop, trim_crop, mask_crop = self.transforms.random_gray_crop(bg_t, trim_t, mask_t)

        final = tch.cat((comp_crop, trim_crop), dim=0)

        return final, mask_crop
    
    def save_debug_tensor(self, t: tch.Tensor, path: Path):
        """
        Saves tensor (C, H, W) as image. 
        Assumes first 3 channels are RGB in [0, 255].
        """
        t = t.detach().cpu().clamp(0, 255).to(tch.uint8)
        img = F.to_pil_image(t[:3])  # только RGB
        img.save(path)
    
    def __len__(self) -> int:
        return len(self.compos_paths)
    
    def __getitem__(self, index) -> tuple[tch.Tensor, tch.Tensor]:
        row = self.compos_paths.iloc[index]
        
        orig = Path(row["original"])
        trim = Path(row["trimap"])
        mask = Path(row["mask"])
        bg = Path(row["background"])
        print(orig)
        compos, mask = self.composite(orig, trim, mask, bg)

        comp_path = Path(__file__).parent / "comp.jpg"
        mask_path = Path(__file__).parent / "mask.jpg"

        self.save_debug_tensor(compos, comp_path)
        self.save_debug_tensor(mask, mask_path)

        return compos, mask
        
        


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
    utl.set_seed()

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
