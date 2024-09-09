import os
import aim
import sys
import copy
import time
import torch
import timeit
import datetime

from pathlib import Path
from torch import Tensor
from torch.backends import cudnn
from torch.nn import Module
from torch.nn.functional import avg_pool2d, interpolate
from torch.optim.optimizer import Optimizer
from torchvision.utils import save_image, make_grid
from ignite.engine import Engine
from ignite.metrics import FID


from data import *
from networks import *
from utils import *
from custom_layers import *
from losses import *

# turn fast mode on
cudnn.benchmark = True

class ProGANTrainer():
    def __init__(
        self,
        base_dir = ".",
        model_dir = "saved_models",
        image_dir = "saved_images",
        epochs: list = [3, 3, 5, 5, 5, 7, 10],
        dataset_name: str = "melbourne-z-top",
        image_size: int = 256,
        input_image_channel: int = 1,
        target_image_channel: int = 3,
        depth: int = 8,
        gen_latent_size: int = 256,
        start_depth: int = 2,
        fade_pct: list = [50 for _ in range(7)],
        batch_sizes: list = [4, 4, 4, 2, 2, 1, 1],
        batch_repeats: int = 1,
        lr_gen = 3e-3,
        lr_dis = 3e-3,
        b1: float = 0.003,
        b2: float = 0.999,
        use_eql: bool = True,
        use_ema: bool = True,
        ema_beta: float = 0.999,
        threads: int = 8,
        sample_every: int = 500,
        sample_num: int = 9,
        sample_grid: bool = True,
        save_every: int = 2,
        loss_fn: str = "wgan_gp",
        calculate_fid_every: int = 2,
        calculate_fid_num_images: int = 1000,
        calculate_fid_batch_size: int = 8,
        pool_size: int = 40,
    ):
        self.name = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        base_dir = Path(base_dir)
        self.model_dir = base_dir / model_dir / self.name
        self.image_dir = base_dir / image_dir / self.name
        self.fid_dir = base_dir / 'fid' / self.name
        self.init_folders()
        
        self.epochs = epochs
        self.depth = depth
        self.gen_latent_size = gen_latent_size
        self.start_depth = start_depth
        self.fade_pct = fade_pct
        self.batch_sizes = batch_sizes
        self.batch_repeats = batch_repeats
        self.use_eql = use_eql
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.optimizers = []
        self.schedulers = []
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.b1 = b1
        self.b2 = b2
        
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.input_image_channel = input_image_channel
        self.target_image_channel = target_image_channel
        if dataset_name == "melbourne-z-top":
            self.input_image_channel = 1
        self.input_shape = (self.input_image_channel, self.image_size, self.image_size)
        self.target_shape = (self.target_image_channel, self.image_size, self.image_size)
        self.threads = threads

        self.set_loss(loss_fn)

        self.sample_every = sample_every
        self.sample_num = sample_num
        self.sample_grid = sample_grid
        
        self.save_every = save_every

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.calculate_fid_batch_size = calculate_fid_batch_size
        self.fid = 0

        self.pool_size = pool_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = aim.Run()

        assert self.depth-1 == len(batch_sizes), "batch_sizes are not compatible with depth"
        assert self.depth-1 == len(epochs), "epochs are not compatible with depth"

    def init_folders(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.fid_dir, exist_ok=True)

    def init_eval_loaders(self):
        if self.dataset_name == "melbourne-top":
            self.sample_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num)
            self.eval_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.calculate_fid_num_images)
        elif self.dataset_name == "melbourne-z-top":
            self.sample_set = MelbourneZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num)
            self.eval_set = MelbourneZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.calculate_fid_num_images)
        elif self.dataset_name == "maps":
            raise ValueError("Maps dataset is not supported yet")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.sample_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.sample_num, shuffle=False)
        self.eval_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.calculate_fid_batch_size, shuffle=False)

    def get_train_loader(self, train_batch_size):
        if self.dataset_name == "melbourne-top":
            self.train_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="train")
        elif self.dataset_name == "melbourne-z-top":
            self.train_set = MelbourneZRGB(dataset=self.dataset_name, image_set="train")
        elif self.dataset_name == "maps":
            raise ValueError("Maps dataset is not supported yet")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return DataLoader(dataset=self.train_set, num_workers=self.threads, batch_size=train_batch_size, shuffle=True)

    def init_generator(self):
        self.gen = Generator(
            depth=self.depth,
            input_channels=self.input_image_channel,
            output_channels=self.target_image_channel,
            use_eql=self.use_eql,
        ).to(self.device)

        print(self.gen)

    def init_discriminator(self):
        self.dis = Discriminator(
            depth=self.depth,
            input_channels=self.target_image_channel,
            latent_size=self.gen_latent_size,
            use_eql=self.use_eql,
        ).to(self.device)

        print(self.dis)

    def init_optimizers(self):
                # create the generator and discriminator optimizers
        self.gen_optim = torch.optim.Adam(
            params=self.gen.parameters(),
            lr=self.lr_gen,
            betas=(self.b1, self.b2),
            eps=1e-8,
        )
        self.dis_optim = torch.optim.Adam(
            params=self.dis.parameters(),
            lr=self.lr_dis,
            betas=(self.b1, self.b2),
            eps=1e-8,
        )

    def set_loss(self, loss_fn):
        if loss_fn == "wgan_gp":
            self.loss_fn = WganGP()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def progressive_downsample_batch(self, real_batch, depth, alpha):
        down_sample_factor = int(2 ** (self.depth - depth))
        prior_downsample_factor = int(2 ** (self.depth - depth + 1))

        ds_real_samples = avg_pool2d(
            real_batch, kernel_size=down_sample_factor, stride=down_sample_factor
        )

        if depth > 2:
            prior_ds_real_samples = interpolate(
                avg_pool2d(
                    real_batch,
                    kernel_size=prior_downsample_factor,
                    stride=prior_downsample_factor,
                ),
                scale_factor=2,
            )
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a linear combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        return real_samples

    def optimize_discriminator(self,
        loss: GANLoss,
        dis_optim: Optimizer,
        input_data: Tensor,
        real_batch: Tensor,
        depth: int,
        alpha: float,
    ) -> float:
        real_samples = self.progressive_downsample_batch(real_batch, depth, alpha)

        # generate a batch of samples
        fake_samples = self.gen(input_data, depth, alpha).detach()
        dis_loss = loss.dis_loss(
            self.dis, real_samples, fake_samples, depth, alpha
        )

        # optimize discriminator
        dis_optim.zero_grad()
        dis_loss.backward()
        if self._check_grad_ok(self.dis):
            dis_optim.step()
        else:
            self.dis_overflow_count += 1

        return dis_loss.item()

    def optimize_generator(
        self,
        loss: GANLoss,
        gen_optim: Optimizer,
        noise: Tensor,
        real_batch: Tensor,
        depth: int,
        alpha: float,
    ) -> float:
        real_samples = self.progressive_downsample_batch(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        gen_loss = loss.gen_loss(self.dis, real_samples, fake_samples, depth, alpha)

        # variation loss criterion
        criterion_tv = TotalVariationLoss().to(self.device)
        tv_loss = criterion_tv(fake_samples) * 0.0001
        gen_loss += tv_loss

        # optimize the generator
        gen_optim.zero_grad()
        gen_loss.backward()
        if self._check_grad_ok(self.gen):
            gen_optim.step()
        else:
            self.gen_overflow_count += 1

        return gen_loss.item()

    def _toggle_all_networks(self, mode="train"):
        for network in (self.gen, self.dis):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    @staticmethod
    def _check_grad_ok(network: Module) -> bool:
        grad_ok = True
        for _, param in network.named_parameters():
            if param.grad is not None:
                param_ok = (
                    torch.sum(torch.isnan(param.grad)) == 0
                    and torch.sum(torch.isinf(param.grad)) == 0
                )
                if not param_ok:
                    grad_ok = False
                    break
        return grad_ok

    def get_save_info(
        self, gen_optim: Optimizer, dis_optim: Optimizer
    ) -> Dict[str, Any]:

        if self.device == torch.device("cpu"):
            generator_save_info = self.gen.get_save_info()
            discriminator_save_info = self.dis.get_save_info()
        else:
            generator_save_info = self.gen.module.get_save_info()
            discriminator_save_info = self.dis.module.get_save_info()
        
        save_info = {
            "generator": generator_save_info,
            "discriminator": discriminator_save_info,
            "gen_optim": gen_optim.state_dict(),
            "dis_optim": dis_optim.state_dict(),
        }
        if self.use_ema:
            save_info["shadow_generator"] = (
                self.gen_shadow.get_save_info()
                if self.device == torch.device("cpu")
                else self.gen_shadow.module.get_save_info()
            )
        return save_info
    
    def sample_images(self, epoch, sample_num, grid=True):
        self.gen.eval()

        fake_images = Tensor([]).to(self.device)
        for i, batch in enumerate(self.sample_dl):
            real_A = batch["A"].to(self.device)
            fake_B = self.gen(real_A)
            fake_B = adjust_dynamic_range(fake_B, drange_in=(-1, 1), drange_out=(0, 1))
            if grid:
                fake_images = torch.cat((fake_images, fake_B), 0)
            else:
                save_image(fake_B, os.path.join(self.image_dir, f"fake_B_{epoch}_{sample_num}_{i}.png"))  # save individual images

        if grid:
            grid = make_grid(fake_images, nrow=3, normalize=True)
            save_image(grid, os.path.join(self.image_dir, f"fake_B_{epoch}_{sample_num}.png"))

        self.gen.train()
    
    def calculate_fid(self, epoch):
        def eval_step(engine, batch):
            return batch

        default_evaluator = Engine(eval_step)
        fid_metric = FID()
        fid_metric.attach(default_evaluator, "fid")
        
        # get evaluation samples
        with torch.no_grad():
            self.gen.eval()
            fake, real = Tensor([]).to(self.device), Tensor([]).to(self.device)
            for eval_batch in self.eval_dl:
                fake = torch.cat((fake, self.gen(eval_batch["A"].to(self.device))), 0)
                real = torch.cat((real, eval_batch["B"].to(self.device)), 0)

            state = default_evaluator.run([[fake, real]])
            self.gen.train()
        
        self.fid = state.metrics["fid"]

        # write to file
        with open(self.fid_dir / f"fid.txt", "a") as f:
            f.write(f"{epoch},{self.fid}\n")

    def train(self) -> None:

        self.init_generator()
        self.init_discriminator()
    
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # initialize the gen_shadow weights equal to the weights of gen
            update_average(self.gen_shadow, self.gen, beta=0)

        self.gen_overflow_count = 0
        self.dis_overflow_count = 0

        self.init_optimizers()
        self.init_eval_loaders()
        
        self._toggle_all_networks("train")
        
        global_step = 0
        print("Starting the training process ... ")
        for current_depth in range(self.start_depth, self.depth + 1):
            current_res = int(2**current_depth)
            print(f"\n\nCurrently working on Depth: {current_depth}")
            print("Current resolution: %d x %d" % (current_res, current_res))
            depth_list_index = current_depth - 2  # start index from 0
            current_batch_size = self.batch_sizes[depth_list_index]
            data = self.get_train_loader(current_batch_size)
            ticker = 1
            
            # because for each depth we can define a different number of epochs
            for epoch in range(1, self.epochs[depth_list_index] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print(f"\nEpoch: {epoch}")
                total_batches = len(data)
                print(f"Total batches: {total_batches}")

                # compute the fader point
                fader_point = int(
                    (self.fade_pct[depth_list_index] / 100)
                    * self.epochs[depth_list_index]
                    * total_batches
                )

                for i, batch in enumerate(data, start=1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    gan_input = batch['A'].to(self.device)
                    images = batch['B'].to(self.device)

                    gen_loss, dis_loss = None, None
                    for _ in range(self.batch_repeats):
                        # optimize the discriminator:
                        dis_loss = self.optimize_discriminator(
                            self.loss_fn, self.dis_optim, gan_input, images, current_depth, alpha
                        )

                        # no idea why this needs to be done after discriminator iteration,
                        # but this is where it is done in the original code
                        if self.use_ema:
                            update_average(
                                self.gen_shadow, self.gen, beta=self.ema_beta
                            )

                        # optimize the generator:
                        gen_loss = self.optimize_generator(
                            self.loss_fn, self.gen_optim, gan_input, images, current_depth, alpha
                        )

                        if self.sample_every and i % self.sample_every == 0:
                            self.sample_images(epoch, i, grid=self.sample_grid)

                        self.logger.track(dis_loss, "dis_loss")
                        self.logger.track(gen_loss, "gen_loss")
                        sys.stdout.write(
                            "\r [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                            % (
                                epoch,
                                self.epochs[self.epochs[depth_list_index]],
                                i,
                                len(data),
                                dis_loss,
                                gen_loss,
                            )
                        )

                    global_step += 1

                    # increment the alpha ticker and the step
                    ticker += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                if (
                    epoch % self.save_every == 0
                    or epoch == 1
                    or epoch == self.epochs[depth_list_index]
                ):
                    save_file = self.model_dir / f"depth_{current_depth}_epoch_{epoch}.bin"
                    torch.save(self.get_save_info(self.gen_optim, self.dis_optim), save_file)

                if self.calculate_fid_every and epoch % self.calculate_fid_every == 0 and epoch != 0:
                    self.calculate_fid(epoch)
                    self.logger.track(self.fid, "fid")

        self._toggle_all_networks("eval")
        print("Training completed ...")

trainer = ProGANTrainer()
trainer.train()
