from pathlib import Path

import torch as tch

from models import AlphaGenerator
import utils as utl
from dataset import CustomDataset


def main() -> None:
    csv_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"

    dataset = CustomDataset(csv_path, train=True)

    sample, mask = dataset[75]

    print(sample.shape)
    print(mask.shape)
    # model = AlphaGenerator()

    # utl.set_seed()

    # x = tch.randn(1, 3, 1024, 1024)

    # with tch.no_grad():
    #     out = model(x)


if __name__ == "__main__":
    main()
