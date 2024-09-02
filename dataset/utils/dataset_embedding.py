import os
import timm
import torch
import numpy as np

from tqdm import tqdm
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

# 1. Randomly select a small batch of images that could fit in memory
# 2. Extract features from the images using the feature extractor and
#    find an average embedding for the dataset
# 3. Compare all images in the dataset with the average embedding

load_dotenv()

class MelbourneRGB(Dataset):
    def __init__(
        self,
        dataset: str = "melbourne-top",
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        assert dataset in ["melbourne-top", "melbourne-z-top"], "Dataset not supported for XYZRGB"

        self.dataset = dataset
        self.base_dir = os.environ.get("DATASET_ROOT") or "."
    
        dataset_path = os.path.join(self.base_dir, self.dataset)
        self.rgb_path = os.path.join(dataset_path, "rgb_data")
        self.rgb_samples = sorted(os.listdir(self.rgb_path)[:max_samples])

        assert len(self.rgb_samples) > 0, "Number of samples should be positive"

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        img_path = os.path.join(self.rgb_path, self.rgb_samples[index])
        img = np.load(img_path)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32)

def get_model(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model

def get_features(model, image):
    return model(image)

def get_loader(max_samples=None, batch_size=4):
    dataset = MelbourneRGB(max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == "__main__":
    model_name = "gluon_inception_v3"
    embedding_shape = (1, 2048)
    model = get_model(model_name)
    loader = get_loader(batch_size=1, max_samples=1000)
    feature_list = torch.zeros(embedding_shape)
    for batch in tqdm(loader):
        features = get_features(model, batch)
        feature_list += features

    avg_dataset_embedding = feature_list / len(loader)
    torch.save(avg_dataset_embedding, f"{model_name}_avg_dataset_embedding.pt")
