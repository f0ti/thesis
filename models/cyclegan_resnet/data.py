import os
import torch
import numpy as np

from typing import Optional, Tuple, Any
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from dotenv import load_dotenv

from utils import adjust_dynamic_range

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

class MelbourneXYZRGB(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne-top",
        image_set: str = "train",
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne-top"], "Dataset not supported for XYZRGB"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform()

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.xyz_path = os.path.join(dataset_path, image_set, "xyz_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.xyz_samples = sorted(os.listdir(self.xyz_path))[:max_samples]
        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]

        assert len(self.rgb_samples) == len(self.xyz_samples), "Number of samples in RGB and XYZ folders do not match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.xyz_path, self.xyz_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        label = np.load(label_path)

        input = self.transform(input)
        label = self.transform(label)

        return {"A": input, "B": label}


class MelbourneZRGB(Dataset):
    """
    The tile dataset that contains the Z and RGB data
    """
    def __init__(
        self,
        dataset: str = "melbourne-z-top",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_dynamic_range: bool = False,
    ):
        super().__init__()

        assert dataset in ["melbourne-z-top"], "Dataset not supported for ZRGB"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform()
        self.adjust_dynamic_range = adjust_dynamic_range

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

        if self.adjust_dynamic_range:
            input = adjust_dynamic_range(input, (0, 1), (-1, 1))
            label = adjust_dynamic_range(label, (0, 1), (-1, 1))

        return {"A": input, "B": label}


class Maps(Dataset):
    def __init__(
        self,
        dataset: str = "maps",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_dynamic_range: bool = False,
    ):
        super().__init__()

        assert dataset in ["maps"], "Dataset not supported for Maps"
        assert image_set in ["train", "test"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Lambda(lambda x: x / 255.0),
                v2.ToDtype(torch.float32),
            ]
        )
        self.resize = v2.Resize(256)

        dataset_path = os.path.join(self.base_dir, self.dataset)
        self.images_path = os.path.join(dataset_path, image_set)
        self.image_samples = sorted(os.listdir(self.images_path))[:max_samples]
        self.adjust_dynamic_range = adjust_dynamic_range

    def __len__(self) -> int:
        return len(self.image_samples)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_path, self.image_samples[index]))
        img = self.transform(img)  # to tensor

        # half left is input, half right is label
        input = img[:, :, : img.shape[2] // 2]
        label = img[:, :, img.shape[2] // 2 :]

        input = self.resize(input)
        label = self.resize(label)

        if self.adjust_dynamic_range:
            input = adjust_dynamic_range(input, (0, 1), (-1, 1))
            label = adjust_dynamic_range(label, (0, 1), (-1, 1))

        return {"A": input, "B": label}
