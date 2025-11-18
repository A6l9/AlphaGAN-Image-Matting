import yaml
from pathlib import Path

from box import Box


train_cfg_path = Path(__file__).parent / "configs" / "train_config.yaml"

with train_cfg_path.open("r") as fp:
    train_cfg = Box(yaml.safe_load(fp))
