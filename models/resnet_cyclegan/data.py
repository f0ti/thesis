import glob
import random
import os
import numpy as np
import torch
from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision.transforms import v2
from dotenv import load_dotenv

load_dotenv()

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
