""" Module for the data loading pipeline for the model to train """

from typing import Any, Optional, Tuple

import os
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from utils import adjust_dynamic_range
from dotenv import load_dotenv

load_dotenv()

class NoOp(object):
    """A NoOp image transform utility. Does nothing, but makes the code cleaner"""

    def __call__(self, whatever: Any) -> Any:
        return whatever

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


def get_transform(
    new_size: Optional[Tuple[int, int]] = None, flip_horizontal: bool = False
):
    """
    obtain the image transforms required for the input data
    Args:
        new_size: size of the resized images (if needed, could be None)
        flip_horizontal: whether to randomly mirror input images during training
    Returns: requested transform object from TorchVision
    """

    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5) if flip_horizontal else NoOp(),
            v2.Resize(new_size) if new_size is not None else NoOp(),
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
        ]
    )


def get_data_loader(
    dataset: Dataset, batch_size: int, num_workers: int = 8
) -> DataLoader:
    """
    generate the data_loader from the given dataset
    Args:
        dataset: Torch dataset object
        batch_size: batch size for training
        num_workers: num of parallel readers for reading the data
    Returns: dataloader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


class MelbourneXYZRGB(Dataset):
    """
    The tile dataset that contains the XYZ and RGB data
    """
    def __init__(
        self,
        dataset: str = "melbourne-top",
        image_set: str = "train",
        transform=get_transform(),
        input_data_range: Tuple[float, float] = (0.0, 1.0),
        output_data_range: Tuple[float, float] = (-1.0, 1.0),
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne-top"], "Dataset not supported"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform
        self.input_data_range = input_data_range
        self.output_data_range = output_data_range

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.xyz_path = os.path.join(dataset_path, image_set, "xyz_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.xyz_samples = sorted(os.listdir(self.xyz_path))[:max_samples]
        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]

        assert len(self.rgb_samples) == len(
            self.xyz_samples
        ), "Number of samples in RGB and XYZ folders do not match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.xyz_path, self.xyz_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        label = np.load(label_path)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        input = adjust_dynamic_range(
            input, drange_in=self.input_data_range, drange_out=self.output_data_range
        )

        label = adjust_dynamic_range(
            label, drange_in=self.input_data_range, drange_out=self.output_data_range
        )

        return {"A": input, "B": label}


class MelbourneZRGB(Dataset):
    """
    The tile dataset that contains the Z and RGB data
    """
    def __init__(
        self,
        dataset: str = "melbourne-z-top",
        image_set: str = "train",
        transform=get_transform(),
        input_data_range: Tuple[float, float] = (0.0, 1.0),
        output_data_range: Tuple[float, float] = (-1.0, 1.0),
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne-top", "melbourne-z-top"], "Dataset not supported"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform
        self.input_data_range = input_data_range
        self.output_data_range = output_data_range

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.z_path = os.path.join(dataset_path, image_set, "z_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.z_samples = sorted(os.listdir(self.z_path))[:max_samples]
        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]

        assert len(self.rgb_samples) == len(
            self.z_samples
        ), "Number of samples in RGB and Z folders do not match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.z_path, self.z_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        label = np.load(label_path)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        input = adjust_dynamic_range(
            input, drange_in=self.input_data_range, drange_out=self.output_data_range
        )

        label = adjust_dynamic_range(
            label, drange_in=self.input_data_range, drange_out=self.output_data_range
        )

        return {"A": input, "B": label}