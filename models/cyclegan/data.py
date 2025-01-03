import os
import torch
import numpy as np

from PIL import Image
from torch import Tensor
from random import shuffle
from typing import Optional, Tuple, Any
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import rgb_to_grayscale
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
        self.transform = get_transform(rotation=True, flip_horizontal=True)

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
        adjust_range: bool = False,
    ):
        super().__init__()

        assert dataset in ["melbourne-z-top"], "Dataset not supported for ZRGB"
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


class EstoniaZIRGB(Dataset):
    """
    The tile dataset that contains the Z+intensity, and RGB data
    """
    def __init__(
        self,
        dataset: str = "estonia-zi",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_range: bool = False,
        normalize_intensity: bool = True,
    ):
        super().__init__()

        assert dataset in ["estonia-zi"], "Dataset not supported for ZIRGB"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform(rotation=True, flip_horizontal=True)
        self.adjust_range = adjust_range
        self.normalize_intensity = normalize_intensity

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.zi_path = os.path.join(dataset_path, image_set, "zi_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.zi_samples = sorted(os.listdir(self.zi_path))[:max_samples]
        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]

        assert len(self.rgb_samples) == len(self.zi_samples), "Nr. of input and target samples don't match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.zi_path, self.zi_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        # normalize the intensity channel by the Z channel, avoid division by zero
        if self.normalize_intensity:
            input[:, :, 0] = input[:, :, 0] / np.maximum(input[:, :, 1], 1e-6)
        label = np.load(label_path)

        if self.transform is not None:
            input = self.transform(input)
            label = self.transform(label)

        if self.adjust_range:
            input = adjust_range(input, (0, 1), (-1, 1))
            label = adjust_range(label, (0, 1), (-1, 1))

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
        self.z_samples = sorted(os.listdir(self.z_path))
        self.rgb_samples = sorted(os.listdir(self.rgb_path))

        # shuffle the samples similarly
        if image_set == "test":
            samples = list(zip(self.z_samples, self.rgb_samples))
            shuffle(samples)
            self.z_samples, self.rgb_samples = zip(*samples)

        self.z_samples = self.z_samples[:max_samples]
        self.rgb_samples = self.rgb_samples[:max_samples]

        assert len(self.rgb_samples) == len(self.z_samples), "Nr. of input and target samples don't match"

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


class EstoniaIRGB(Dataset):
    """
    The tile dataset that contains the Z+intensity, and RGB data
    """
    def __init__(
        self,
        dataset: str = "estonia-i",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_range: bool = False,
    ):
        super().__init__()

        assert dataset in ["estonia-i"], "Dataset not supported for ZIRGB"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform(rotation=True, flip_horizontal=True)
        self.adjust_range = adjust_range

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.i_path = os.path.join(dataset_path, image_set, "i_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")
        self.i_samples = sorted(os.listdir(self.i_path))
        self.rgb_samples = sorted(os.listdir(self.rgb_path))

        # shuffle the samples similarly
        if image_set == "test":
            samples = list(zip(self.i_samples, self.rgb_samples))
            shuffle(samples)
            self.i_samples, self.rgb_samples = zip(*samples)

        self.i_samples = self.i_samples[:max_samples]
        self.rgb_samples = self.rgb_samples[:max_samples]

        assert len(self.rgb_samples) == len(self.i_samples), "Nr. of input and target samples don't match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.i_path, self.i_samples[index])
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


class Maps(Dataset):
    def __init__(
        self,
        dataset: str = "maps",
        image_set: str = "train",
        max_samples: Optional[int] = None,
        adjust_range: bool = False,
    ):
        super().__init__()

        assert dataset in ["maps", "graymaps"], "Dataset not supported for Maps"
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
        self.adjust_range = adjust_range

    def __len__(self) -> int:
        return len(self.image_samples)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_path, self.image_samples[index]))
        img = self.transform(img)

        # half left is street, half right is satellite
        input = img[:, :, img.shape[2] // 2 :]
        if self.dataset == "graymaps":
            input = rgb_to_grayscale(input)
        label = img[:, :, : img.shape[2] // 2]

        input = self.resize(input)
        label = self.resize(label)

        if self.adjust_range:
            input = adjust_range(input, (0, 1), (-1, 1))
            label = adjust_range(label, (0, 1), (-1, 1))

        return {"A": input, "B": label}


class TileSelector(Dataset):
    def __init__(
        self,
        dataset: str = "estonia-i",
        image_set: str = "test",
        max_samples: Optional[int] = None,
        adjust_range: bool = False,
        tile_id: Optional[str] = None,
    ):
        super().__init__()

        assert dataset in ["estonia-i"], "Dataset not supported for TileSelector"
        assert image_set in ["test", "val"]

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = get_transform(rotation=True, flip_horizontal=True)
        self.adjust_range = adjust_range
        self.tile_id = tile_id
        self.data_type = dataset.split("-")[-1]

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.coo_path = os.path.join(dataset_path, image_set, f"{self.data_type}_data")
        self.rgb_path = os.path.join(dataset_path, image_set, "rgb_data")

        self.xyz_samples = sorted(os.listdir(self.coo_path))
        self.rgb_samples = sorted(os.listdir(self.rgb_path))

        if self.tile_id is not None:
            self.xyz_samples = [sample for sample in self.xyz_samples if self.tile_id in sample]
            self.rgb_samples = [sample for sample in self.rgb_samples if self.tile_id in sample]

        self.xyz_samples = self.xyz_samples[:max_samples]
        self.rgb_samples = self.rgb_samples[:max_samples]

        assert len(self.rgb_samples) == len(self.xyz_samples), "Number of samples in RGB and XYZ folders do not match"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.coo_path, self.xyz_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        file_id = self.xyz_samples[index].split(".")[0]

        input = np.load(input_path)
        label = np.load(label_path)

        input = self.transform(input)
        label = self.transform(label)

        return {"A": input, "B": label, "tile_id": file_id}
