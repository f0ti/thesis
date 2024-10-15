import os
import aim
import sys
import fire
import torch
import datetime

from pathlib import Path
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore, SSIM
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from model import *
from loss import *
from data import *

class Trainer:
    def __init__(
        self,
        base_dir = ".",
        model_dir = "saved_models",
        image_dir = "saved_images",
        load_from_model_dir: str = "2024-10-11_18-18-30_graymaps_resnet9",
        load_from_model_epoch: int = 48,
        adjust_model_shape: bool = False,
        epochs: int = 15,
        epochs_decay: int = 35,
        dataset_name: str = "estonia-i",
        image_size: int = 256,
        input_image_channel: int = 1,
        target_image_channel: int = 3,
        adjust_range: bool = True,
        generator_type: str = "resnet9",
        use_dropout: bool = False,
        batch_size: int = 2,
        lr_gen = 2e-4,
        lr_disc = 2e-4,
        lr_policy: str = "linear",
        b1: float = 0.5,
        b2: float = 0.999,
        threads: int = 8,
        sample_every: int = 1000,
        sample_num: int = 9,
        sample_grid: bool = True,
        save_every: int = 2,
        loss_fns: list = ["gan", "cycle", "identity", "ssim", "tv", "lpips"],
        loss_weights: list = [1.0, 10.0, 0.5, 0.5, 1.0, 5.0],
        eval_fid: bool = True,
        eval_is: bool = False,
        eval_every: int = 2,
        eval_samples: int = 2000,
        eval_batch_size: int = 8,
        pool_size: int = 50,
    ):
        date_now = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.name = f"{date_now}_{dataset_name}_{generator_type}"

        base_dir = Path(base_dir)
        self.model_dir = base_dir / model_dir / self.name
        self.image_dir = base_dir / image_dir / self.name
        self.eval_dir = base_dir / 'eval' / self.name
        self.load_from_model_dir = None
        if load_from_model_dir:
            self.load_from_model_dir = base_dir / model_dir / load_from_model_dir
            self.load_from_model_epoch = load_from_model_epoch
            print("Loading from model directory:", self.load_from_model_dir)
        self.init_folders()
        self.adjust_model_shape = adjust_model_shape

        self.total_epochs = epochs + epochs_decay
        self.epochs = epochs
        self.epochs_decay = epochs_decay
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.lr_policy = lr_policy
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.b1 = b1
        self.b2 = b2
        self.threads = threads
        self.generator_type = generator_type
        self.use_dropout = use_dropout
        self.pool_size = pool_size
        
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights

        self.sample_every = sample_every
        self.sample_num = sample_num
        self.sample_grid = sample_grid
        
        self.save_every = save_every

        self.eval_every = eval_every
        self.eval_samples = eval_samples
        self.eval_batch_size = eval_batch_size
        self.eval_fid = eval_fid
        self.eval_is = eval_is
        assert (self.eval_fid or self.eval_is), "At least one of FID or IS should be enabled"

        self.image_size = image_size
        self.input_image_channel = input_image_channel
        self.target_image_channel = target_image_channel
        self.adjust_range = adjust_range

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = aim.Run(log_system_params=True)
    
    def init_GAN(self):
        # Models
        self.init_generator()
        self.init_discriminator()

        # Losses
        self.init_losses()

        # Optimizers
        self.init_optimizers()
        
        # Learning rate update schedulers
        self.init_schedulers()

        # Buffers
        self.init_buffers()

    def init_generator(self):
        if self.generator_type == "resnet6":
            self.G_AB = GeneratorResNet(self.input_image_channel, self.target_image_channel, n_blocks=6, use_dropout=self.use_dropout).to(self.device)
            self.G_BA = GeneratorResNet(self.target_image_channel, self.input_image_channel, n_blocks=6, use_dropout=self.use_dropout).to(self.device)
        elif self.generator_type == "resnet9":
            self.G_AB = GeneratorResNet(self.input_image_channel, self.target_image_channel, n_blocks=9, use_dropout=self.use_dropout).to(self.device)
            self.G_BA = GeneratorResNet(self.target_image_channel, self.input_image_channel, n_blocks=9, use_dropout=self.use_dropout).to(self.device)
        elif self.generator_type == "resnet18":
            self.G_AB = GeneratorResNet(self.input_image_channel, self.target_image_channel, n_blocks=18, use_dropout=self.use_dropout).to(self.device)
            self.G_BA = GeneratorResNet(self.target_image_channel, self.input_image_channel, n_blocks=18, use_dropout=self.use_dropout).to(self.device)
        elif self.generator_type == "unet":
            self.G_AB = GeneratorUNet(self.input_image_channel, self.target_image_channel, num_downs=8).to(self.device)
            self.G_BA = GeneratorUNet(self.target_image_channel, self.input_image_channel, num_downs=8).to(self.device)
        else:
            raise ValueError(f"Unknown generator type: {self.generator_type}")

        if self.load_from_model_dir:
            print("Loading models from", self.load_from_model_dir)
            self.load_model(self.G_AB, self.load_from_model_dir / f"G_AB_{self.load_from_model_epoch}.pth", "G_AB")
            self.load_model(self.G_BA, self.load_from_model_dir / f"G_BA_{self.load_from_model_epoch}.pth", "G_BA")
            print(f"Loaded generators from {self.load_from_model_dir}")
        else:
            init_weights(self.G_AB)
            init_weights(self.G_BA)

    def init_discriminator(self):
        self.D_A = PatchDiscriminator(self.input_image_channel).to(self.device)
        self.D_B = PatchDiscriminator(self.target_image_channel).to(self.device)

        if self.load_from_model_dir:
            self.load_model(self.D_A, self.load_from_model_dir / f"D_A_{self.load_from_model_epoch}.pth", "D_A")
            self.load_model(self.D_B, self.load_from_model_dir / f"D_B_{self.load_from_model_epoch}.pth", "D_B")
            print(f"Loaded discriminators from {self.load_from_model_dir}")
        else:
            init_weights(self.D_A)
            init_weights(self.D_B)
    
    def init_folders(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.eval_dir,  exist_ok=True)

    def init_optimizers(self):
        self.optim_G_AB = torch.optim.Adam(self.G_AB.parameters(), lr=self.lr_disc, betas=(self.b1, self.b2))
        self.optim_G_BA = torch.optim.Adam(self.G_BA.parameters(), lr=self.lr_disc, betas=(self.b1, self.b2))
        self.optim_D_A  = torch.optim.Adam(self.D_A.parameters(), lr=self.lr_gen, betas=(self.b1, self.b2))
        self.optim_D_B  = torch.optim.Adam(self.D_B.parameters(), lr=self.lr_gen, betas=(self.b1, self.b2))
        self.optimizers = [self.optim_G_AB, self.optim_G_BA, self.optim_D_A, self.optim_D_B]

    def init_schedulers(self):
        self.schedulers = [get_scheduler(optimizer, self.epochs, self.epochs_decay) for optimizer in self.optimizers]

    def init_losses(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        # self.criterion_identity = torch.nn.L1Loss()
        # self.criterion_ssim = SSIMLoss().to(self.device)
        self.criterion_lpips = LPIPS().to(self.device)
        # self.criterion_tv = TVLoss().to(self.device)
        self.lambda_GAN = self.loss_weights[self.loss_fns.index("gan")]
        self.lambda_cycle = self.loss_weights[self.loss_fns.index("cycle")]
        # self.lambda_identity = self.loss_weights[self.loss_fns.index("identity")]
        # self.lambda_ssim = self.loss_weights[self.loss_fns.index("ssim")]
        self.lambda_lpips = self.loss_weights[self.loss_fns.index("lpips")]
        # self.lambda_tv = self.loss_weights[self.loss_fns.index("tv")]

    def init_buffers(self):
        # buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer(self.pool_size)
        self.fake_B_buffer = ReplayBuffer(self.pool_size)

    def init_data(self):
        if self.dataset_name == "melbourne-top":
            self.train_set  = MelbourneXYZRGB(dataset=self.dataset_name, image_set="train")
            self.sample_set = MelbourneXYZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num)
            self.eval_set   = MelbourneXYZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples)
        elif self.dataset_name == "melbourne-z-top":
            self.train_set  = MelbourneZRGB(dataset=self.dataset_name, image_set="train", adjust_range=self.adjust_range)
            self.sample_set = MelbourneZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num, adjust_range=self.adjust_range)
            self.eval_set   = MelbourneZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples, adjust_range=self.adjust_range)
        elif self.dataset_name in ["maps", "graymaps"]:
            self.train_set  = Maps(dataset=self.dataset_name, image_set="train", adjust_range=self.adjust_range)
            self.sample_set = Maps(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num, adjust_range=self.adjust_range)
            self.eval_set   = Maps(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-z":
            self.train_set  = EstoniaZRGB(dataset=self.dataset_name, image_set="train", adjust_range=self.adjust_range)
            self.sample_set = EstoniaZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num, adjust_range=self.adjust_range)
            self.eval_set   = EstoniaZRGB(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-i":
            self.train_set  = EstoniaIRGB(dataset=self.dataset_name, image_set="train", adjust_range=self.adjust_range)
            self.sample_set = EstoniaIRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num, adjust_range=self.adjust_range)
            self.eval_set   = EstoniaIRGB(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples, adjust_range=self.adjust_range)
        elif self.dataset_name == "estonia-zi":
            self.train_set  = EstoniaZIRGB(dataset=self.dataset_name, image_set="train", adjust_range=self.adjust_range)
            self.sample_set = EstoniaZIRGB(dataset=self.dataset_name, image_set="test", max_samples=self.sample_num, adjust_range=self.adjust_range)
            self.eval_set   = EstoniaZIRGB(dataset=self.dataset_name, image_set="test", max_samples=self.eval_samples, adjust_range=self.adjust_range)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.dataloader = DataLoader(dataset=self.train_set, num_workers=self.threads, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.sample_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.sample_num, shuffle=False, drop_last=True)
        self.eval_dl = DataLoader(dataset=self.sample_set, num_workers=self.threads, batch_size=self.eval_batch_size, shuffle=False, drop_last=True)

    def load_model(self, model, model_dir, model_name):
        loaded_model = torch.load(model_dir)
        if self.adjust_model_shape:
            print("Adjusting model shape")
            if model_name == "G_AB":    # if model is G_AB duplicate model.1.weight
                loaded_model['model.1.weight'] = torch.cat((loaded_model['model.1.weight'], loaded_model['model.1.weight']), 1)
                self.match_weights(model, loaded_model, model_name, 1)
            elif model_name == "G_BA":  # if model is G_BA duplicate model.26.weight 
                loaded_model['model.26.weight'] = torch.cat((loaded_model['model.26.weight'], loaded_model['model.26.weight']), 1)
                self.match_weights(model, loaded_model, model_name, 1)
            elif model_name == "D_A":   # if model is D_A duplicate model.0.weight
                loaded_model['model.0.weight'] = torch.cat((loaded_model['model.0.weight'], loaded_model['model.0.weight']), 1)
                self.match_weights(model, loaded_model, model_name, 0)
            else:  # if model is D_B just load the model
                model.load_state_dict(loaded_model)
        else:
            model.load_state_dict(loaded_model)

    def match_weights(self, model, loaded_model, model_name, idx):
        if model.state_dict()[f'model.{idx}.weight'].shape == loaded_model[f'model.{idx}.weight'].shape:
            print(f"Model {model_name} input layers now match")
            return True
        return False

    def save_configs(self):
        with open(self.model_dir / "configs.txt", "w") as f:
            f.write(str(self.__dict__))

    def track(self, value, name):
        if self.logger is None:
            return
        self.logger.track(value, name = name)

    def _toggle_networks(self, mode="train"):
        for network in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    def print_networks(self):
        for network in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            print(network)

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        self.logger.track(lr, "learning_rate")
        print('learning rate %.5f -> %.5f' % (old_lr, lr))

    def save_model(self, epoch):
        torch.save(self.G_AB.state_dict(), os.path.join(self.model_dir, f"G_AB_{str(epoch).zfill(2)}.pth"))
        torch.save(self.G_BA.state_dict(), os.path.join(self.model_dir, f"G_BA_{str(epoch).zfill(2)}.pth"))
        torch.save(self.D_A.state_dict(), os.path.join(self.model_dir, f"D_A_{str(epoch).zfill(2)}.pth"))
        torch.save(self.D_B.state_dict(), os.path.join(self.model_dir, f"D_B_{str(epoch).zfill(2)}.pth"))
    
    def sample_images(self, epoch, sample_num, grid=True):
        self.G_AB.eval()

        real_images, fake_images = Tensor([]).to(self.device), Tensor([]).to(self.device)
        for i, batch in enumerate(self.sample_dl):
            real_A = batch["A"].to(self.device)
            fake_B = self.G_AB(real_A)
            if self.adjust_range:
                fake_B = adjust_range(fake_B, (-1, 1), (0, 1))
            if grid:
                fake_images = torch.cat((fake_images, fake_B), 0)
                if epoch == 0:
                    real_images = torch.cat((real_images, batch["B"].to(self.device)), 0)
            else:
                save_image(fake_B, os.path.join(self.image_dir, f"fake_B_{str(epoch).zfill(4)}_{sample_num}_{i}.png"))

        if grid:
            fake_grid = make_grid(fake_images, nrow=3, normalize=True)
            if epoch == 0:
                real_grid = make_grid(real_images, nrow=3, normalize=True)
                save_image(real_grid, os.path.join(self.image_dir, f"real_B_{str(epoch).zfill(4)}_{sample_num}.png"))
            save_image(fake_grid, os.path.join(self.image_dir, f"fake_B_{str(epoch).zfill(4)}_{sample_num}.png"))

        self.G_AB.train()
    
    def calculate_eval(self, epoch):
        def eval_step(_, batch):
            return batch

        default_evaluator = Engine(eval_step)
        if self.eval_is:
            is_metric = InceptionScore(device=self.device, output_transform=lambda x: x[0])
            is_metric.attach(default_evaluator, "is")
        if self.eval_fid:
            fid_metric = FID(device=self.device)
            fid_metric.attach(default_evaluator, "fid")

        # get evaluation samples
        with torch.no_grad():
            self.G_AB.eval()
            fake, real = Tensor([]).to(self.device), Tensor([]).to(self.device)
            for eval_batch in self.eval_dl:
                fake = torch.cat((fake, self.G_AB(eval_batch["A"].to(self.device))), 0)
                real = torch.cat((real, eval_batch["B"].to(self.device)), 0)

            state = default_evaluator.run([[fake, real]])
            self.G_AB.train()
        
        self.fid_score = state.metrics["fid"] if self.eval_fid else 0.0
        self.is_score = state.metrics["is"] if self.eval_is else 0.0

        self.logger.track(self.fid_score, "fid")
        self.logger.track(self.is_score, "is")

        # write to file
        with open(self.eval_dir / "evals.txt", "a") as f:
            f.write(f"Epoch: {str(epoch).zfill(2)}, FID: {self.fid_score}, IS: {self.is_score}\n")

    def _zero_grad(self):
        self.optim_G_AB.zero_grad()
        self.optim_G_BA.zero_grad()
        self.optim_D_A.zero_grad()
        self.optim_D_B.zero_grad()

    def train(self):
        print(f"Starting training {self.name}")
        self.init_GAN()
        self.init_data()
        self.save_configs()
        self._toggle_networks("train")

        # adversarial ground truths
        valid_patch_A = torch.ones((self.batch_size, *self.D_A.output_shape), requires_grad=False).cuda()
        fake_patch_A  = torch.zeros((self.batch_size, *self.D_A.output_shape), requires_grad=False).cuda()
        valid_patch_B = torch.ones((self.batch_size, *self.D_B.output_shape), requires_grad=False).cuda()
        fake_patch_B  = torch.zeros((self.batch_size, *self.D_B.output_shape), requires_grad=False).cuda()
        
        for epoch in range(self.total_epochs):
            for i, batch in enumerate(self.dataloader):
                real_A = batch["A"].cuda()
                real_B = batch["B"].cuda()

                self._zero_grad()

                # train generators

                # Identity loss (use only for 3>3)
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
                loss_cycle = (loss_cycle_A + loss_cycle_B) * 0.5

                # Total Variation loss
                # loss_tv = self.criterion_tv(fake_B)

                # Structural Similarity loss
                # loss_ssim_G_A = self.criterion_ssim(fake_A, real_A)
                # loss_ssim_G_B = self.criterion_ssim(fake_B, real_B)
                # loss_ssim = (loss_ssim_G_A + loss_ssim_G_B) * 0.5

                # Learned Perceptual Image Patch Similarity loss
                loss_lpips = self.criterion_lpips(fake_B, real_B)

                # Total loss (generator)
                loss_G =  self.lambda_GAN * loss_adv_G
                loss_G += self.lambda_cycle * loss_cycle
                # loss_G += self.lambda_ssim * loss_ssim
                loss_G += self.lambda_lpips * loss_lpips
                # loss_G += self.lambda_identity * loss_identity
                # loss_G += loss_tv * self.lambda_tv
                
                loss_G.backward()
                self.optim_G_AB.step()
                self.optim_G_BA.step()

                # train discriminator A
                loss_real = self.criterion_GAN(self.D_A(real_A), valid_patch_A)  # real loss
                fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake_patch_A)  # fake loss
                loss_D_A = (loss_real + loss_fake) * 0.5  # total loss

                # train discriminator B
                loss_real = self.criterion_GAN(self.D_B(real_B), valid_patch_B)  # real loss
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake_patch_B)  # fake loss
                loss_D_B = (loss_real + loss_fake) * 0.5  # total loss

                loss_D = (loss_D_A + loss_D_B) * 0.5

                loss_D.backward()
                self.optim_D_A.step()
                self.optim_D_B.step()

                if self.sample_every and i % self.sample_every == 0 and i != 0:
                    self.sample_images(epoch, i, grid=self.sample_grid)
                
                if self.save_every and epoch % self.save_every == 0 and epoch != 0:
                    self.save_model(epoch)

                self.logger.track(loss_D.item(), "loss_D")
                self.logger.track(loss_G.item(), "loss_G")
                self.logger.track(loss_adv_G.item(), "loss_adv_G")
                self.logger.track(loss_cycle.item(), "loss_cycle")
                # self.logger.track(loss_ssim.item(), "loss_ssim")
                self.logger.track(loss_lpips.item(), "loss_lpips")
                # self.logger.track(loss_identity.item(), "loss_identity")
                # self.logger.track(loss_tv.item(), "loss_tv")

                sys.stdout.write(
                    "\r [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, lpips: %f]"
                    % (
                        epoch+1,
                        self.total_epochs,
                        i,
                        len(self.dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_adv_G.item(),
                        loss_cycle.item(),
                        # loss_ssim.item(),
                        loss_lpips.item(),
                        # loss_tv.item()
                    )
                )
            
            if self.eval_every and epoch % self.eval_every == 0 and epoch != 0:
                self.calculate_eval(epoch)

            self.update_learning_rate()

if __name__ == '__main__':
    fire.Fire(Trainer)
