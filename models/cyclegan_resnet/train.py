import os
import itertools
import datetime
import time
import torch
import aim
import sys

from math import floor
from pathlib import Path
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

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
        loss_fns: list = ["gan", "cycle", "identity"],
        loss_weights: list = [10.0, 10.0, 0.5],
        calculate_fid_every = None,
        calculate_fid_num_images = 1000,
        clear_fid_cache = False,
        pool_size: int = 50,
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
            self.input_image_channel = 1
        self.input_shape = (self.input_image_channel, self.image_size, self.image_size)
        self.target_shape = (self.target_image_channel, self.image_size, self.image_size)

        self.pool_size = pool_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = aim.Run()

    @property
    def checkpoint_num(self):
        return floor(self.epochs // self.save_every)
    
    def init_GAN(self):
        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(self.input_image_channel, self.target_image_channel, n_blocks=6).to(self.device)
        self.G_BA = GeneratorResNet(self.target_image_channel, self.input_image_channel, n_blocks=6).to(self.device)
        self.D_A = PatchDiscriminator(self.input_image_channel).to(self.device)
        self.D_B = PatchDiscriminator(self.target_image_channel).to(self.device)

        # self.print_networks()

        init_weights(self.G_AB)
        init_weights(self.G_BA)
        init_weights(self.D_A)
        init_weights(self.D_B)

        # Losses
        self.init_losses()

        # Optimizers
        self.init_optimizers()
        
        # Learning rate update schedulers
        self.init_schedulers()

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer(self.pool_size)
        self.fake_B_buffer = ReplayBuffer(self.pool_size)

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

    def init_losses(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        # self.criterion_tv = TVLoss()
        self.lambda_GAN = self.loss_weights[self.loss_fns.index("gan")]
        self.lambda_cycle = self.loss_weights[self.loss_fns.index("cycle")]
        self.lambda_identity = self.loss_weights[self.loss_fns.index("identity")]
        # self.lambda_tv = self.loss_weights[self.loss_fns.index("tv")]

    def set_data_src(self):
        if self.dataset_name == "melbourne-top":
            self.train_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="train")
        elif self.dataset_name == "melbourne-z-top":
            self.train_set = MelbourneZRGB(dataset=self.dataset_name, image_set="train")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.dataloader = DataLoader(dataset=self.train_set, num_workers=self.threads, batch_size=self.batch_size, shuffle=True)

    def track(self, value, name):
        if self.logger is None:
            return
        self.logger.track(value, name = name)

    def _toggle_all_networks(self, mode="train"):
        for network in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    def print_networks(self):
        print(self.G_AB)
        print(self.G_BA)
        print(self.D_A)
        print(self.D_B)

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

        # adversarial ground truths
        valid_patch_A = torch.ones((self.batch_size, *self.D_A.output_shape), requires_grad=False).cuda()
        fake_patch_A = torch.zeros((self.batch_size, *self.D_A.output_shape), requires_grad=False).cuda()
        valid_patch_B = torch.ones((self.batch_size, *self.D_B.output_shape), requires_grad=False).cuda()
        fake_patch_B = torch.zeros((self.batch_size, *self.D_B.output_shape), requires_grad=False).cuda()

        print(f"valid_patch_A: {valid_patch_A.shape}, fake_patch_A: {fake_patch_A.shape}")
        print(f"valid_patch_B: {valid_patch_B.shape}, fake_patch_B: {fake_patch_B.shape}")
        
        prev_time = time.time()
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.dataloader):
                real_A = batch["A"].cuda()
                real_B = batch["B"].cuda()

                self.optim_D.zero_grad()
                self.optim_G.zero_grad()

                # train generator

                # Identity loss
                # loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
                # loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
                # loss_identity = (loss_id_A + loss_id_B) * 0.5

                # Adversarial loss
                fake_B = self.G_AB(real_A)  # what generator thinks is B
                loss_adv_AB = self.criterion_GAN(self.D_B(fake_B), valid_patch_B)
                fake_A = self.G_BA(real_B)  # what generator thinks is A
                loss_adv_BA = self.criterion_GAN(self.D_A(fake_A), valid_patch_A)
                loss_adv_G = (loss_adv_AB + loss_adv_BA) * 0.5

                # Cycle loss
                recov_A = self.G_BA(fake_B)  # what generator thinks is A after converting B
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                recov_B = self.G_AB(fake_A)  # what generator thinks is B after converting A
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total Variation loss
                # loss_tv = criterion_tv(fake_A)  # do it only for A>B

                # Total loss (generator)
                loss_G =  loss_adv_G
                loss_G += self.lambda_cycle * loss_cycle
                # loss_G += self.lambda_identity * loss_identity
                # loss_G += loss_tv * opt.lambda_tv
                
                loss_G.backward()
                self.optim_G.step()

                # train discriminator A
                loss_real = self.criterion_GAN(self.D_A(real_A), valid_patch_A)  # real loss
                fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake_patch_A)  # fake loss
                loss_D_A = (loss_real + loss_fake) / 2  # total loss

                # train discriminator B
                loss_real = self.criterion_GAN(self.D_B(real_B), valid_patch_B)  # real loss
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake_patch_B)  # fake loss
                loss_D_B = (loss_real + loss_fake) / 2  # total loss

                loss_D = (loss_D_A + loss_D_B) / 2

                loss_D.backward()
                self.optim_D.step()

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = self.epochs * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # if self.sample_every and epoch % self.sample_every == 0 and epoch != 0:
                #     self.sample_images(epoch)
                
                if self.save_every and epoch % self.save_every == 0 and epoch != 0:
                    self.save_model(epoch)
                
                # if self.calculate_fid_every and epoch % self.calculate_fid_every == 0 and epoch != 0:
                #     self.calculate_fid(epoch)

                self.logger.track(loss_D.item(), "loss_D")
                self.logger.track(loss_G.item(), "loss_G")
                self.logger.track(loss_adv_G.item(), "loss_adv_G")
                self.logger.track(loss_cycle.item(), "loss_cycle")
                # self.logger.track(loss_identity.item(), "loss_identity")
                # self.logger.track(loss_tv.item(), "loss_tv")
                sys.stdout.write("\r Time left: %s" % str(time_left))

            # Update learning rates
            self.update_learning_rate()

trainer = Trainer()
trainer.train()
