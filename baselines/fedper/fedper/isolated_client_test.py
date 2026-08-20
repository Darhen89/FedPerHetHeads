from pathlib import Path
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedper.utils import (
    set_num_classes,
    )
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg = set_num_classes(cfg)
    print("Config: ",cfg)

main()