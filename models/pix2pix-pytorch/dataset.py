import os
import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from config import *
from PIL import Image


class TileDataset(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne",
        image_set: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        print(dataset)
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
        label = transforms.ToTensor()(label)

        if self.image_set == "train":
            return image, label
        else:
            return image

train_dataset = TileDataset(dataset="melbourne", image_set="train", max_samples=10)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(next(iter(train_dataloader))[0])
print(next(iter(train_dataloader))[1])
