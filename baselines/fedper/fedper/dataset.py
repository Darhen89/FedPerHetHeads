"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from fedper.dataset_preparation import (corel5k_preprocess,)

# working dir is two up
WORKING_DIR = Path(__file__).resolve().parent.parent
FL_BENCH_ROOT = WORKING_DIR.parent

sys.path.append(FL_BENCH_ROOT.as_posix())


def dataset_main(config: dict) -> None:
    """Prepare the dataset."""
    dataset_name = config["name"].lower()
    dataset_folder = Path(WORKING_DIR, "datasets")
    dataset_root = Path(dataset_folder, dataset_name)

    if not os.path.isdir(dataset_root):
        os.makedirs(dataset_root)

    if dataset_name.lower() == "corel5k":
        return corel5k_preprocess(dataset_root, config)
    else:
        raise RuntimeError("Please implement the dataset preparation for your dataset.")
