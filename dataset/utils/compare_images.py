import os
import sys
import timm
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from dotenv import load_dotenv

# 1. Randomly select a small batch of images that could fit in memory
# 2. Extract features from the images using the feature extractor and
#    find an average embedding for the dataset
# 3. Compare all images in the dataset with the average embedding

load_dotenv()

def show_rgb(img):
    plt.imshow(img)
    plt.show()

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

    def filename(self, index):
        return self.rgb_samples[index]

    def __len__(self) -> int:
        return len(self.rgb_samples)

    def __getitem__(self, index):
        img_path = os.path.join(self.rgb_path, self.rgb_samples[index])
        img = np.load(img_path)
        img = img.transpose(2, 0, 1)
        filename = self.rgb_samples[index]
        return {
            "img": torch.tensor(img, dtype=torch.float32),
            "filename": filename
        }

def get_dataset_embedding(file_name: str):
    return torch.load(file_name)

def get_loader(max_samples=None):
    return DataLoader(MelbourneRGB(max_samples=max_samples), shuffle=True)

def get_model(model_name):
    with torch.no_grad():
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        return model

def get_features(model, image):
    return model(image)

if __name__ == "__main__":
    embedding_filename = "gluon_inception_v3_avg_dataset_embedding.pt"
    dataset_embedding = get_dataset_embedding(embedding_filename)
    loader = get_loader()
    model = get_model("gluon_inception_v3")
    threshold = 0.97
    filtered_dir = f"filtered_out_{threshold}"
    os.makedirs(filtered_dir, exist_ok=True)
    nr_filtered = 0
    # clear blacklist
    with open(f"blacklist_{threshold}.txt", "w") as f:
        f.write("")

    # compare all images in the dataset with the average embedding
    for i, batch in enumerate(loader, 1):
        batch_embeddings = get_features(model, batch['img'])
        similarity = F.cosine_similarity(dataset_embedding, batch_embeddings)
        sys.stdout.write("\r [%d/%d] \t %s" % (i, len(loader), similarity))
        
        if similarity < threshold:
            nr_filtered += 1
            print(f"Filtered out image: {batch['filename'][0]}. Total filtered out: {nr_filtered}")
            # write in blacklist.txt
            with open(f"blacklist_{threshold}.txt", "a") as f:
                f.write(f"{batch['filename'][0]}\n")
            
            # save the image as png
            img = batch['img'].numpy().transpose(0, 2, 3, 1)[0]
            plt.imsave(f"{filtered_dir}/{batch['filename'][0]}.png", img)

    print(f"Filtered out images: {nr_filtered}")
