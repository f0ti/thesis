import os
import numpy as np
import itertools
import datetime
import time
import sys
import wandb
import torch
import aim

from math import floor
from pathlib import Path
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from utils import *
from model import *
from loss import *
from data import *
from config import *

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

class Trainer():
    def __init__(
        self,
        name = "default",
        base_dir = ".",
        model_dir = "saved_models",
        image_dir = "images",
        epochs: int = 10,
        dataset_name: str = "melbourne-z-top",
        image_size: int = 256,
        input_image_channel: int = 3,
        target_image_channel: int = 3,
        batch_size: int = 1,
        lr_gen = 5e-5,
        lr_disc = 4e-4,
        target_lr_gen = 1e-5,
        target_lr_disc = 1e-4,
        lr_decay_span = 5,
        b1: float = 0.5,
        b2: float = 0.999,
        threads: int = 8,
        sample_every: int = 0,
        save_every: int = 2,
        n_residual_blocks: int = 9,
        loss_fns: list = ["gan", "cycle", "identity", "tv", "ssim", "sdi"],
        loss_weights: list = [1.0, 10.0, 5.0, 0.1, 1.0, 1.0],
        calculate_fid_every = None,
        calculate_fid_num_images = 1000,
        clear_fid_cache = False,
        log: bool = True
    ):
        self.name = name,

        base_dir = Path(base_dir)
        self.model_dir = base_dir / model_dir
        self.image_dir = base_dir / image_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.model_dir / name / '.config.json'
        self.init_folders()
        
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.optimizers = []
        self.schedulers = []
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.target_lr_gen = target_lr_gen
        self.target_lr_disc = target_lr_disc
        self.lr_decay_span = lr_decay_span
        self.b1 = b1
        self.b2 = b2
        self.threads = threads
        self.n_residual_blocks = n_residual_blocks
        
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights

        self.sample_every = sample_every
        self.save_every = save_every

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.image_size = image_size
        self.input_image_channel = input_image_channel
        self.target_image_channel = target_image_channel
        if dataset_name == "melbourne-z-top":
            self.target_image_channel = 1
        self.input_shape = (self.input_image_channel, self.image_size, self.image_size)
        self.target_shape = (self.target_image_channel, self.image_size, self.image_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = aim.Session(experiment=name) if log else None

    @property
    def checkpoint_num(self):
        return floor(self.epochs // self.save_every)
    
    def init_GAN(self):
        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(self.input_shape, self.target_shape, self.n_residual_blocks).to(self.device)
        self.G_BA = GeneratorResNet(self.input_shape, self.target_shape, self.n_residual_blocks).to(self.device)
        self.D_A = Discriminator(self.target_shape).to(self.device)
        self.D_B = Discriminator(self.input_shape).to(self.device)

        # Optimizers
        self.init_optimizers()
        
        # Learning rate update schedulers
        self.init_schedulers()

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def init_folders(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.fid_dir, exist_ok=True)

    def init_optimizers(self):
        self.optim_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.lr_disc, betas=(self.b1, self.b2))
        self.optim_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=self.lr_gen, betas=(self.b1, self.b2))
        self.optimizers.append(self.optim_G)
        self.optimizers.append(self.optim_D)

    def init_schedulers(self):
        D_decay_fn = lambda i: max(1 - i / self.lr_decay_span, 0) + (self.target_lr_disc / self.lr_disc) * min(i / self.lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / self.lr_decay_span, 0) + (self.target_lr_gen / self.lr_gen) * min(i / self.lr_decay_span, 1)
        self.lr_sched_D = LambdaLR(self.optim_D, D_decay_fn)
        self.lr_sched_G = LambdaLR(self.optim_G, G_decay_fn)
        self.schedulers.append(self.lr_sched_D)
        self.schedulers.append(self.lr_sched_G)

    def set_data_src(self):
        if self.dataset_name == "melbourne-top":
            self.train_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="train")
        elif self.dataset_name == "melbourne-z-top":
            self.train_set = MelbourneZRGB(dataset=self.dataset_name, image_set="train")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        dataloader = DataLoader(dataset=self.train_set, num_workers=self.threads, batch_size=self.batch_size, shuffle=True)
        self.loader = cycle(dataloader)

    def _toggle_all_networks(self, mode="train"):
        for network in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")
 
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.4f -> %.4f' % (old_lr, lr))

    def save_model(self, epoch):
        torch.save(self.G_AB.state_dict(), os.path.join(self.model_dir, f"G_AB_{epoch}.pth"))
        torch.save(self.G_BA.state_dict(), os.path.join(self.model_dir, f"G_AB_{epoch}.pth"))

    def train(self):
        self.init_GAN()
        self.set_data_src()

        self._toggle_all_networks("train")

        total_dis_loss = torch.tensor(0.).to(self.device)
        total_gen_loss = torch.tensor(0.).to(self.device)

        batch_size = self.batch_size

        for epoch in range(self.epochs):
            for i, batch in enumerate(self.loader):
                epoch_start_time = time.time()
                iter_start_time = time.time()
                # Set model input
                real_A = batch["A"].cuda()
                real_B = batch["B"].cuda()

                # Adversarial ground truths
                valid = torch.ones((real_A.size(0), *self.D_A.output_shape)).cuda()
                fake = torch.zeros((real_A.size(0), *self.D_A.output_shape)).cuda()
                self.optim_D.zero_grad()
                self.optim_G.zero_grad()

                self.train_step()

                self.update_learning_rates()

                if self.sample_every and epoch % self.sample_every == 0 and epoch != 0:
                    self.sample_images(epoch)
                
                if self.save_every and epoch % self.save_every == 0 and epoch != 0:
                    self.save_model(epoch)
                
                if self.calculate_fid_every and epoch % self.calculate_fid_every == 0 and epoch != 0:
                    self.calculate_fid(epoch)


        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
    
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total Variation loss
        # loss_tv = criterion_tv(fake_A)  # do it only for the forward pass A -> B

        # Structural Similarity loss
        loss_ssim = criterion_ssim(fake_B, real_B)  # compare what the generator_AB generated with the real B (RGB images)

        # Structural Distortion Index loss
        loss_sdi = criterion_sdi(fake_B, real_B)  # compare what the generator_AB generated with the real B (RGB images)

        # Total losses (generator)
        loss_G =  loss_GAN
        loss_G += opt.lambda_cyc * loss_cycle
        loss_G += opt.lambda_id * loss_identity
        # loss_G += loss_tv * opt.lambda_tv
        loss_G += loss_sdi * opt.lambda_sdi
        loss_G += loss_ssim * opt.lambda_ssim

        loss_G.backward()
        optimizer_G.step()

        loss_G = optimize_generator()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dl) + i
        batches_left = opt.epochs * len(train_dl) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if opt.wb:
            wandb.log({"Loss_D": loss_D.item(), "Loss_G": loss_G.item(), "ssim": loss_ssim.item(), "sdi": loss_sdi.item(), "adv": loss_GAN.item(), "cycle": loss_cycle.item(), "identity": loss_identity.item()})

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, sdi: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.epochs,
                i,
                len(train_dl),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_sdi.item(),
                # loss_ssim.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if opt.sample_interval:
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


# def sample_images(batches_done):
#     """Saves a generated sample from the test set"""
#     if opt.height_data:
#         test_set = MelbourneZRGB(dataset=opt.dataset_name, image_set="test")
#     else:
#         test_set = MelbourneXYZRGB(dataset=opt.dataset_name, image_set="test")
#     test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)
#     imgs = next(iter(test_dl))
#     G_AB.eval()
#     G_BA.eval()
#     real_A = imgs["A"].cuda()
#     fake_B = G_AB(real_A)
#     real_B = imgs["B"].cuda()
#     fake_A = G_BA(real_B)
#     # Arange images along x-axis
#     real_A = make_grid(real_A, nrow=5, normalize=True)
#     real_B = make_grid(real_B, nrow=5, normalize=True)
#     fake_A = make_grid(fake_A, nrow=5, normalize=True)
#     fake_B = make_grid(fake_B, nrow=5, normalize=True)
#     # Arange images along y-axis
#     image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
#     save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

# ----------
#  Training
# ----------