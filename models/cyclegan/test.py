import os
from numpy import real
from scipy.__config__ import show
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from dotenv import load_dotenv

from data import *
from utils import *
from model import *

load_dotenv()

class Sampler():
    def __init__(
        self,
        run_name: str = "2024-09-19_20-20-04_estonia_resnet9",
        model_epoch: str = "32",
        dataset_name: str = "estonia",
        threads: int = 8,
        generator_type: str = "resnet9",
        generator_mode: str = "AB",
        input_channel: int = 1,
        target_channel: int = 3,
        num_samples: int = 9,
        save_images: bool = False
    ) -> None:
        
        assert generator_mode in ("AB", "BA")
        model_dir = Path("saved_models")
        if generator_mode == "AB":
            self.model_path = model_dir / run_name / f"G_AB_{model_epoch}.pth"
        else:
            self.model_path = model_dir / run_name / f"G_BA_{model_epoch}.pth"
        self.dataset_name = dataset_name
        self.threads = threads
        self.generator_type = generator_type
        self.generator_mode = generator_mode
        self.input_channel = input_channel
        self.target_channel = target_channel
        self.num_samples = num_samples
        self.save_images = save_images
        if self.save_images:
            os.makedirs("sampled_images", exist_ok=True)
            self.image_dir = Path("sampled_images") / f"{run_name}_G_{generator_mode}_{model_epoch}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_generator(self):
        if self.generator_type == "resnet6":
            if self.generator_mode == "AB":
                self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=6).to(self.device)
            else:
                self.gen = GeneratorResNet(self.target_channel, self.input_channel, n_blocks=6).to(self.device)
        elif self.generator_type == "resnet9":
            if self.generator_mode == "AB":
                self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=9).to(self.device)
            else:
                self.gen = GeneratorResNet(self.target_channel, self.input_channel, n_blocks=9).to(self.device)
        elif self.generator_type == "unet_upconv":
            if self.generator_mode == "AB":
                self.gen = GeneratorUNet(self.input_channel, self.target_channel, 8, upsample=False).to(self.device)
            else:
                self.gen = GeneratorUNet(self.target_channel, self.input_channel, 8, upsample=False).to(self.device)
        elif self.generator_type == "unet_upsample":
            if self.generator_mode == "AB":
                self.gen = GeneratorUNet(self.input_channel, self.target_channel, 8, upsample=True).to(self.device)
            else:
                self.gen = GeneratorUNet(self.target_channel, self.input_channel, 8, upsample=True).to(self.device)
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

        self.gen.load_state_dict(torch.load(self.model_path))
        self.gen.eval()
        print(f"Loaded generator from path {self.model_path}")

    def init_dataloader(self):
        if self.dataset_name == "melbourne-top":
            self.sample_set = MelbourneXYZRGB(image_set="test", max_samples=self.num_samples)
        elif self.dataset_name == "melbourne-z-top":
            self.sample_set = MelbourneZRGB(image_set="test", max_samples=self.num_samples)
        elif self.dataset_name == "maps":
            self.sample_set = Maps(image_set="test", max_samples=self.num_samples)
        elif self.dataset_name == "graymaps":
            self.sample_set = GrayMaps(image_set="test", max_samples=self.num_samples)
        elif self.dataset_name == "estonia":
            self.sample_set = EstoniaZRGB(image_set="test", max_samples=self.num_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.sample_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.num_samples, shuffle=True)

    def sample_images(self, num_samples, grid=True):
        real_images, fake_images = Tensor([]).to(self.device), Tensor([]).to(self.device)
        for i, batch in enumerate(self.sample_dl):
            # change input to be label if generator is BA
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            fake_B = self.gen(real_A)
            if grid:
                fake_images = torch.cat((fake_images, fake_B), 0)
                real_images = torch.cat((real_images, real_B), 0)
            else:
                # needs fix
                show_z(real_A)
                show_rgb(fake_B)
                show_rgb(real_B)
                if self.save_images:
                    uid = np.random.randint(0, 1000)
                    save_image(real_A, os.path.join(self.image_dir, f"real_A_{num_samples}_{uid}.png"))  # input
                    save_image(real_B, os.path.join(self.image_dir, f"real_B_{num_samples}_{uid}.png"))  # target
                    save_image(fake_B, os.path.join(self.image_dir, f"fake_B_{num_samples}_{uid}.png"))  # output

        if grid:
            fake_grid = make_grid(fake_images, nrow=3, normalize=True)
            show_grid(fake_grid)
            real_grid = make_grid(real_images, nrow=3, normalize=True)
            show_grid(real_grid)
            if self.save_images:
                save_image(real_grid, os.path.join(self.image_dir, f"real_B_{num_samples}.png"))
                save_image(fake_grid, os.path.join(self.image_dir, f"fake_B_{num_samples}.png"))

if __name__ == "__main__":
    sampler = Sampler()
    sampler.init_generator()
    sampler.init_dataloader()
    sampler.sample_images(num_samples=sampler.num_samples, grid=True)
