import os
import numpy as np
import torch

from torch import Tensor
from typing import Optional, Callable, Any, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import v2
from dotenv import load_dotenv

from utils import adjust_range

load_dotenv()

class NoOp(object):
    """A NoOp image transform utility. Does nothing, but makes the code cleaner"""

    def __call__(self, whatever: Any) -> Any:
        return whatever

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class RandomizedRotation(object):
    """ Randomly rotate the input image by a given angle and probability """
    
    def __init__(self, angle: Tuple[int, int] = (-90, 90), p: float = 0.2):
        self.angle = angle
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if np.random.rand() < self.p:
            return v2.RandomRotation(self.angle)(img)
        else:
            return img


def get_transform(
    new_size: Optional[Tuple[int, int]] = None, rotation: bool = False,
    flip_horizontal: bool = False
):
    """
    obtain the image transforms required for the input data
    Args:10
        new_size: size of the resized images (if needed, could be None)
        flip_horizontal: whether to randomly mirror input images during training
    Returns: requested transform object from TorchVision
    """
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5) if flip_horizontal else NoOp(), 
            RandomizedRotation() if rotation else NoOp(),
            # v2.Resize(new_size) if new_size is not None else NoOp(),
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
        ]
    )

class RGBTileDataset(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne-top",
        image_set: str = "train",
        transform: Optional[Callable] = None,
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

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.xyz_path = os.path.join(dataset_path, image_set, "xyz_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.xyz_samples = sorted(os.listdir(self.xyz_path))[:max_samples]
        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]

        assert len(self.rgb_samples) == len(self.xyz_samples), "Number of samples in RGB and XYZ folders do not match"

        self.input_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )

        self.label_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.ToDtype(torch.float32),
            ]
        )

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.xyz_path, self.xyz_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        label = np.load(label_path)

        input = self.input_transforms(input)
        label = self.label_transforms(label)

        return {"A": input, "B": label}


class EstoniaZRGB(Dataset):
    """
    The tile dataset that contains the Z+intensity, and RGB data
    """
    def __init__(
        self,
        dataset: str = "estonia-z",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_range: bool = False
    ):
        super().__init__()

        assert dataset in ["estonia-z"], "Dataset not supported for ZIRGB"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform(rotation=True, flip_horizontal=True)
        self.adjust_range = adjust_range

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

        if self.adjust_range:
            input = adjust_range(input, (0, 1), (-1, 1))
            label = adjust_range(label, (0, 1), (-1, 1))

        return {"A": input, "B": label}
