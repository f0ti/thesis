import os
import torch
import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms
from config import *
from torchvision.transforms import v2
from PIL import Image


class DMTTileDataset(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne",
        image_set: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne"], "Dataset not supported"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = DATA_DIR
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.rgb_path = os.path.join(dataset_path, "A", image_set)
        self.xyz_path = os.path.join(dataset_path, "B", image_set)

        self.rgb_samples = os.listdir(self.rgb_path)
        self.xyz_samples = os.listdir(self.xyz_path)

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        image_path = os.path.join(self.rgb_path, self.rgb_samples[index])
        label_path = os.path.join(self.xyz_path, self.xyz_samples[index])

        image = Image.open(image_path)
        label = np.load(label_path).astype(np.float32)

        image = transforms.ToTensor()(image)[:3, :, :]
        image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        label = transforms.ToTensor()(label)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if self.image_set == "train":
            return image, label
        else:
            return image


class RGBTileDataset(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne",
        image_set: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne"], "Dataset not supported"
        assert image_set in ["train", "test", "val"]

        self.dataset = dataset
        self.base_dir = DATA_DIR
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.rgb_path = os.path.join(dataset_path, "A", image_set)
        self.xyz_path = os.path.join(dataset_path, "B", image_set)

        self.rgb_samples = os.listdir(self.rgb_path)
        self.xyz_samples = os.listdir(self.xyz_path)

        self.input_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.label_transforms = v2.Compose(
            [
                v2.ToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        input_path = os.path.join(self.xyz_path, self.xyz_samples[index])
        label_path = os.path.join(self.rgb_path, self.rgb_samples[index])

        input = np.load(input_path)
        label = Image.open(label_path).convert("RGB")

        input = self.input_transforms(input)
        label = self.label_transforms(label)

        if self.image_set == "train":
            return input, label
        else:
            return label


# train_dataset = RGBTileDataset(dataset="melbourne", image_set="train", max_samples=10)
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# print(next(iter(train_dataloader))[0][0][0][0])
# print(next(iter(train_dataloader))[1][0][0][0])
