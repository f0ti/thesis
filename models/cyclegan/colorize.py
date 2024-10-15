import os
import fire
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dotenv import load_dotenv

from data import *
from utils import *
from model import *

load_dotenv()

class Colorizer():
    def __init__(
        self,
        model_name: str = "2024-10-13_01-27-59_estonia-i_resnet9",
        model_epoch: str = "48",
        dataset_name: str = "estonia-i",
        threads: int = 8,
        generator_type: str = "resnet9",
        tile_id: str = "589551",
        input_channel: int = 1,
        target_channel: int = 3,
        num_samples: Optional[int] = None,
        adjust_range: bool = True,
    ):
        model_dir = Path("saved_models")
        self.model_path = model_dir / model_name / f"G_AB_{model_epoch}.pth"

        colorized_dir = Path("colorized_images")
        self.image_dir = colorized_dir / tile_id
        self.fake_B_dir = self.image_dir / "fake_B"
        self.real_A_dir = self.image_dir / "real_A"
        self.real_B_dir = self.image_dir / "real_B"
        os.makedirs(self.fake_B_dir, exist_ok=True)
        os.makedirs(self.real_A_dir, exist_ok=True)
        os.makedirs(self.real_B_dir, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.threads = threads
        self.generator_type = generator_type
        self.input_channel = input_channel
        self.target_channel = target_channel
        self.adjust_range = adjust_range
        self.tile_id = tile_id        
        self.num_samples = num_samples

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_generator()
        self.init_dataloader()

    def init_generator(self):
        if self.generator_type == "resnet6":
            self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=6).to(self.device)
        elif self.generator_type == "resnet9":
            self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=9).to(self.device)
        elif self.generator_type == "unet":
            self.gen = GeneratorUNet(self.input_channel, self.target_channel, num_downs=8).to(self.device)
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

        self.gen.load_state_dict(torch.load(self.model_path))
        self.gen.eval()
        print(f"Loaded generator from path {self.model_path}")

    def init_dataloader(self):
        self.sample_set = TileSelector(self.dataset_name, max_samples=self.num_samples, tile_id=self.tile_id, adjust_range=self.adjust_range)
        self.sample_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.num_samples, shuffle=False)
        print(f"Loaded dataloader with {len(self.sample_set)} samples")

    def log_file(self):
        # write a log file
        message =  "Colorized with model: %s\n" % self.model_path
        message += "Dataset: %s\n" % self.dataset_name
        message += "Generator type: %s\n" % self.generator_type
        message += "Tile ID: %s\n" % self.tile_id
        message += "Number of samples: %d\n" % len(self.sample_set)
        message += "Adjust range: %s\n" % self.adjust_range
        with open(os.path.join(self.image_dir, "log.txt"), "w") as f:
            f.write(message)

    def colorize(self):
        for batch in self.sample_dl:
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            fake_B = self.gen(real_A)

            if self.adjust_range:
                fake_B = adjust_range(fake_B, (-1, 1), (0, 1))

            assert self.tile_id in batch["tile_id"], f"Expected tile_id {self.tile_id}, got {batch['tile_id']}"

            print(f"Colorizing tile {batch['tile_id']}...")
            save_image(real_A, os.path.join(self.real_A_dir, f"{batch['tile_id']}.png"))  # input
            save_image(real_B, os.path.join(self.real_B_dir, f"{batch['tile_id']}.png"))  # target
            save_image(fake_B, os.path.join(self.fake_B_dir, f"{batch['tile_id']}.png"))  # output

        self.log_file()

if __name__ == "__main__":
    fire.Fire(Colorizer)
