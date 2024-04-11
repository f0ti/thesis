import os
import shutil
import importlib
from pathlib import Path
from typing import Union

# inspired by: https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py

DATASET_PATH = "../../dataset"


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

    # for image_set in ["train", "test"]:
    #     image_set_path = os.path.join(dataset_path, image_set)
    #     if not os.path.exists(image_set_path):
    #         os.makedirs(image_set_path)
    #     for sample in dataset.paths[image_set]:
    #         sample_name = os.path.basename(sample)
    #         sample_path = os.path.join(image_set_path, sample_name)
    #         if not os.path.exists(sample_path):
    #             shutil.copy(sample, sample_path)


make_dataset()
