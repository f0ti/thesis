import glob
import random
import os
import numpy as np
import torch
from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

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
        self.base_dir = "dataset"
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform

        dataset_path = os.path.join(self.base_dir, self.dataset)

        self.rgb_path = os.path.join(dataset_path, "A", image_set)
        self.xyz_path = os.path.join(dataset_path, "B", image_set)

        self.rgb_samples = sorted(os.listdir(self.rgb_path))[:max_samples]
        self.xyz_samples = sorted(os.listdir(self.xyz_path))[:max_samples]

        assert len(self.rgb_samples) == len(self.xyz_samples), "Number of samples in RGB and XYZ folders do not match"

        self.input_transforms = v2.Compose(
            [
                v2.ToTensor(),
                # normalize between 0 and 1
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
