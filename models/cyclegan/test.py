import os
import fire
from sympy import false
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.ssim import SSIM
from dotenv import load_dotenv

from data import *
from utils import *
from model import *

load_dotenv()

class Sampler():
    def __init__(
        self,
        run_name: str = "2024-10-15_11-38-12_estonia-z_resnet9",
        model_epoch: str = "12",
        dataset_name: str = "estonia-z",
        threads: int = 8,
        generator_type: str = "resnet9",
        generator_mode: str = "AB",
        input_channel: int = 1,
        target_channel: int = 3,
        num_samples: int = 9,
        adjust_range: bool = True,
        save_images: bool = True,
        merge_fake_real: bool = True,
        make_grid: bool = True
    ):
        model_dir = Path("saved_models")
        assert generator_mode in ["AB", "BA"], f"Unknown generator mode: {generator_mode}"
        self.model_path = model_dir / run_name / f"G_{generator_mode}_{model_epoch}.pth"
        
        self.dataset_name = dataset_name
        self.threads = threads
        self.generator_type = generator_type
        self.generator_mode = generator_mode
        self.input_channel = input_channel
        self.target_channel = target_channel
        self.adjust_range = adjust_range
        
        self.num_samples = num_samples
        self.save_images = save_images
        self.merge_fake_real = merge_fake_real
        self.make_grid = make_grid
        if self.save_images:
            self.image_dir = Path("sampled_images") / f"{run_name}_G_{generator_mode}_{model_epoch}_{dataset_name}"
            os.makedirs(self.image_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_generator()
        self.init_dataloader()

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
        elif self.generator_type == "unet":
            if self.generator_mode == "AB":
                self.gen = GeneratorUNet(self.input_channel, self.target_channel, num_downs=8).to(self.device)
            else:
                self.gen = GeneratorUNet(self.target_channel, self.input_channel, num_downs=8).to(self.device)
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
            self.sample_set = Maps(dataset="maps", image_set="test", max_samples=self.num_samples)
        elif self.dataset_name == "graymaps":
            self.sample_set = Maps(dataset="graymaps", image_set="test", max_samples=self.num_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-z":
            self.sample_set = EstoniaZRGB(image_set="test", max_samples=self.num_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-i":
            self.sample_set = EstoniaIRGB(image_set="test", max_samples=self.num_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-zi":
            self.sample_set = EstoniaZIRGB(image_set="test", max_samples=self.num_samples, adjust_range=self.adjust_range)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.sample_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.num_samples, shuffle=True)

    def sample_grid(self):
        input_images, real_images, fake_images = Tensor([]).to(self.device), Tensor([]).to(self.device), Tensor([]).to(self.device)
        for batch in self.sample_dl:
            uid = identifier()
            # change input to be label if generator is BA
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            fake_B = self.gen(real_A)
            input_images = torch.cat((input_images, real_A), 0)
            fake_images = torch.cat((fake_images, fake_B), 0)
            real_images = torch.cat((real_images, real_B), 0)

        fake_grid = make_grid(fake_images, nrow=3, normalize=True)
        real_grid = make_grid(real_images, nrow=3, normalize=True)
        show_grid(fake_grid)
        show_grid(real_grid)
        if self.save_images:
            if self.merge_fake_real:
                fake_real_grid = torch.cat((fake_grid, real_grid), 2)
                save_image(fake_real_grid, os.path.join(self.image_dir, f"fake_real_{self.num_samples}_{uid}.png"))
            else:                
                save_image(real_grid, os.path.join(self.image_dir, f"real_A_{self.num_samples}_{uid}.png"))
                save_image(fake_grid, os.path.join(self.image_dir, f"fake_B_{self.num_samples}_{uid}.png"))

    # TODO: fix shape of input images
    def sample_one(self):
        batch = next(iter(self.sample_dl))
        uid = identifier()
        # change input to be label if generator is BA
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)
        fake_B = self.gen(real_A)
        show_z(real_A)
        show_rgb(fake_B)
        show_rgb(real_B)
        if self.save_images:
            save_image(real_A, os.path.join(self.image_dir, f"real_A_{uid}.png"))  # input
            save_image(real_B, os.path.join(self.image_dir, f"real_B_{uid}.png"))  # target
            save_image(fake_B, os.path.join(self.image_dir, f"fake_B_{uid}.png"))  # output


class Evaluator():
    def __init__(
        self,
        base_dir = Path("."),
        model_name: str = "2024-10-11_18-18-30_graymaps_resnet9",
        model_epoch: str = "48",
        model_mode: str = "AB",
        generator_type: str = "resnet9",
        dataset_name: str = "estonia-i",
        threads: int = 8,
        input_channel: int = 1,
        target_channel: int = 3,
        eval_fid: bool = True,
        eval_is: bool = True,
        eval_ssim: bool = True,
        eval_samples: int = 2000,
        eval_batch_size: int = 8,
        ) -> None:

        model_dir = Path("saved_models")
        self.model_path = model_dir / f"{model_name}/G_{model_mode}_{model_epoch}.pth"
        self.eval_fid = eval_fid
        self.eval_is = eval_is
        self.eval_ssim = eval_ssim
        self.eval_samples = eval_samples
        self.eval_batch_size = eval_batch_size
        self.eval_dir = base_dir / 'eval' / model_name

        # TODO: get this from the model name
        self.dataset_name = dataset_name
        self.threads = threads
        self.model_mode = model_mode
        self.generator_type = generator_type
        self.input_channel = input_channel
        self.target_channel = target_channel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_generator()
        self.init_dataloader()

    def init_generator(self):
        if self.generator_type == "resnet6":
            if self.model_mode == "AB":
                self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=6).to(self.device)
            else:
                self.gen = GeneratorResNet(self.target_channel, self.input_channel, n_blocks=6).to(self.device)
        elif self.generator_type == "resnet9":
            if self.model_mode == "AB":
                self.gen = GeneratorResNet(self.input_channel, self.target_channel, n_blocks=9).to(self.device)
            else:
                self.gen = GeneratorResNet(self.target_channel, self.input_channel, n_blocks=9).to(self.device)
        elif self.generator_type == "unet":
            if self.model_mode == "AB":
                self.gen = GeneratorUNet(self.input_channel, self.target_channel, num_downs=8).to(self.device)
            else:
                self.gen = GeneratorUNet(self.target_channel, self.input_channel, num_downs=8).to(self.device)
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

    def init_dataloader(self):
        if self.dataset_name == "melbourne-top":
            self.eval_set = MelbourneXYZRGB(image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "melbourne-z-top":
            self.eval_set = MelbourneZRGB(image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "maps":
            self.eval_set = Maps(dataset="maps", image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "graymaps":
            self.eval_set = Maps(dataset="graymaps", image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "estonia-z":
            self.eval_set = EstoniaZRGB(image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "estonia-i":
            self.eval_set = EstoniaIRGB(image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "estonia-zi":
            self.eval_set = EstoniaZIRGB(image_set="test", max_samples=self.eval_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.eval_dl = DataLoader(dataset=self.eval_set, num_workers=self.threads, batch_size=self.eval_batch_size, shuffle=False, drop_last=True)

        self.gen.load_state_dict(torch.load(self.model_path))
        self.gen.eval()
        print(f"Loaded generator from path {self.model_path}")

    def eval(self):
        def eval_step(_, batch):
            return batch

        default_evaluator = Engine(eval_step)
        if self.eval_is:
            is_metric = InceptionScore(device=self.device, output_transform=lambda x: x[0])
            is_metric.attach(default_evaluator, "is")
        if self.eval_fid:
            fid_metric = FID(device=self.device)
            fid_metric.attach(default_evaluator, "fid")
        if self.eval_ssim:
            ssim_metric = SSIM(data_range=2.0, device=self.device)
            ssim_metric.attach(default_evaluator, "ssim")

        # get evaluation samples
        with torch.no_grad():
            self.gen.eval()
            fake, real = Tensor([]).to(self.device), Tensor([]).to(self.device)
            for eval_batch in self.eval_dl:
                fake = torch.cat((fake, self.gen(eval_batch["A"].to(self.device))), 0)
                real = torch.cat((real, eval_batch["B"].to(self.device)), 0)

        print(fake.shape, real.shape)
        state = default_evaluator.run([[fake, real]])
        
        self.fid_score = state.metrics["fid"] if self.eval_fid else 0.0
        self.is_score = state.metrics["is"] if self.eval_is else 0.0
        self.ssim_score = state.metrics["ssim"] if self.eval_ssim else 0.0

        # write to file
        with open(self.eval_dir / "evals.txt", "a") as f:
            log = f"FID: {self.fid_score:.3f}, IS: {self.is_score}, SSIM: {self.ssim_score}\n"
            f.write(log)
            print(log)

if __name__ == "__main__":
    fire.Fire(Sampler)
    # fire.Fire(Evaluator)
