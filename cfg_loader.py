import yaml
from pathlib import Path

from box import Box


cfg_path = Path(__file__).parent / "configs" / "config.yaml"

with cfg_path.open("r") as fp:
    cfg = Box(yaml.safe_load(fp))
