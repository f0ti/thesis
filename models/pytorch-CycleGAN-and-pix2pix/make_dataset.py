import os
import shutil
import importlib
from pathlib import Path
from typing import Union

# inspired by: https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py

DATASET_PATH = "../../dataset"
DATASET_TARGET = "data/melbourne"


def validate_dataset(dataset_path: str) -> bool:
    # check if rgb_data and xyz_data are in the dataset
    required_dirs = {"rgb_data", "xyz_data"}
    dirs = set(x.name for x in os.scandir(dataset_path) if x.is_dir())
    
    # check if train and test are in required dirs
    required_subdirs = {"train", "test"}
    rgb_subdirs = set(x.name for x in os.scandir(os.path.join(dataset_path, "rgb_data")) if x.is_dir())
    xyz_subdirs = set(x.name for x in os.scandir(os.path.join(dataset_path, "xyz_data")) if x.is_dir())

    return required_dirs.issubset(dirs) and required_subdirs.issubset(rgb_subdirs) and required_subdirs.issubset(xyz_subdirs)


def make_dataset(dataset_root: Union[str, Path] = DATASET_PATH, dataset_name: str = "melbourne"):
    dataset_path = os.path.join(os.path.expanduser(dataset_root), dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
    
    assert validate_dataset(dataset_path), "Required dirs are missing. Please refer to the documentation."

    for d in ["A", "B"]:
        for split in ["train", "test"]:
            print(f"Copying {d} {split} data")
            os.makedirs(os.path.join(DATASET_TARGET, d, split), exist_ok=True)

            src = "rgb_data" if d == "A" else "xyz_data"
            shutil.copytree(os.path.join(dataset_path, src, split), os.path.join(DATASET_TARGET, d, split), dirs_exist_ok=True)

make_dataset()
