import os
from torch import Tensor
import numpy as np
import ctypes
from tqdm import tqdm
from tile import Tile
from helper import download_and_extract_archive, DATASET_URL
from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_DIR = "../tiles_data"


class TileDataset(Dataset):
    def __init__(
        self,
        root: str = "melbourne",
        download: bool = False,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert root in ["melbourne"], "Dataset not supported"
        assert image_set in ["train", "test", "val"]

        self.root = root
        self.base_dir = DATASET_DIR
        self.image_set = image_set
        self.max_samples = max_samples
        self.transform = transform

        self.IMAGE_SHAPE = (256, 256, 3)

        dataset_path = os.path.join(self.base_dir, self.root)

        if download:
            download_and_extract_archive(DATASET_URL[self.root], self.root)

        if not os.path.isdir(dataset_path):
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )
        
        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")

        self.paths = {
            "train": [],
            "test": [],
        }

        self.paths["train"] = list(
            map(lambda x: os.path.join(train_path, x), os.listdir(train_path))
        )

        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )

        self.samples = self.paths[image_set]
        self.num_samples = len(self.samples)

        if self.max_samples:
            self.num_samples = min(self.max_samples, self.num_samples)
            self.samples = self.samples[: self.num_samples]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):

        sample = Tile(self.samples[index])
        image = np.asarray(sample.colors, dtype=np.int8).reshape(sample.width, sample.height, 3)
        
        if self.image_set != "test":
            label = np.asarray(sample.xyz, dtype=np.double).reshape(sample.width, sample.height, 3)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        if self.image_set == "train":
            return (image, label)

        else:
            return (image,)
