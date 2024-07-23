""" Module implementing ProGAN which is trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import copy
import datetime
import dis
import time
import timeit
import wandb
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from torch.nn.functional import avg_pool2d, interpolate
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image

from custom_layers import update_average
from data import get_data_loader
from losses import CycleGANLoss, GANLoss, WganGP, TotalVariationLoss, SSIMLoss, SDILoss
from networks import Discriminator, Generator
from utils import adjust_dynamic_range


class ProGAN:
    def __init__(
        self,
        gen: Generator,
        dis: Discriminator,
        device=torch.device("cuda"),
        use_ema: bool = False,
        ema_beta: float = 0.999,
    ):
        assert gen.depth == dis.depth, (
            f"Generator and Discriminator depths are not compatible. "
            f"GEN_Depth: {gen.depth}  DIS_Depth: {dis.depth}"
        )
        self.gen = gen.to(device)
        self.dis = dis.to(device)
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.depth = gen.depth
        self.device = device

        # if code is to be run on GPU, we can use DataParallel:
        if device == torch.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        print(f"Generator Network: {self.gen}")
        print(f"Discriminator Network: {self.dis}")

        # instead of using the optimized parameters from the final training
        # iteration (parameter update step) as the final parameters for the
        # model, the exponential moving average of the parameters over the
        # course of all the training iterations are used.
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # initialize the gen_shadow weights equal to the weights of gen
            update_average(self.gen_shadow, self.gen, beta=0)

        # counters to maintain generator and discriminator gradient overflows
        self.gen_overflow_count = 0
        self.dis_overflow_count = 0

    def progressive_downsample_batch(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        Args:
            real_batch: batch of real samples
            depth: depth at which training is going on
            alpha: current value of the fader alpha

        Returns: modified real batch of samples

        """
        # downsample the real_batch for the given depth
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

        # real samples are a linear combination of
        # ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        return real_samples

    def optimize_discriminator(
        self,
        loss: GANLoss,
        dis_optim: Optimizer,
        input_data: Tensor,
        real_batch: Tensor,
        depth: int,
        alpha: float,
    ) -> float:
        """
        performs a single weight update step on discriminator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            dis_optim: discriminator optimizer
            noise: input noise for sample generation
            real_batch: real samples batch
            depth: current depth of optimization
            alpha: current alpha for fade-in

        Returns: discriminator loss value
        """
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
        """
        performs a single weight update step on generator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            gen_optim: generator optimizer
            noise: input noise for sample generation
            real_batch: real samples batch
            depth: current depth of optimization
            alpha: current alpha for fade-in

        Returns: generator loss value
        """
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

    @staticmethod
    def create_grid(
        samples: Tensor,
        scale_factor: int,
        img_file: Path,
    ) -> None:
        """
        utility function to create a grid of GAN samples
        Args:
            samples: generated samples for feedback
            scale_factor: factor for upscaling the image
            img_file: name of file to write
        Returns: None (saves a file)
        """
        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)
        
        samples = adjust_dynamic_range(
            samples, drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
        )

        # save the images
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))), padding=0)

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

    def train(
        self,
        dataset: Dataset,
        epochs: List[int],
        batch_sizes: List[int],
        fade_in_percentages: List[int],
        loss_fn: GANLoss = WganGP(),
        batch_repeats: int = 4,
        gen_learning_rate: float = 0.003,
        dis_learning_rate: float = 0.003,
        num_samples: int = 16,
        start_depth: int = 2,
        num_workers: int = 3,
        feedback_factor: int = 100,
        save_dir=Path("./saved_models"),
        checkpoint_factor: int = 10,
        wb_mode: bool = True,
    ):
        """
        # TODO implement support for conditional GAN here
        Utility method for training the ProGAN.
        Note that you don't have to necessarily use this method. You can use the
        optimize_generator and optimize_discriminator and define your own
        training routine
        Args:
            dataset: object of the dataset used for training.
                     Note that this is not the dataloader (we create dataloader in this
                     method since the batch_sizes for resolutions can be different)
            epochs: list of number of epochs to train the network for every resolution
            batch_sizes: list of batch_sizes for every resolution
            fade_in_percentages: list of percentages of epochs per resolution
                                used for fading in the new layer not used for
                                first resolution, but dummy value is still needed
            loss_fn: loss function used for training
            batch_repeats: number of iterations to perform on a single batch
            gen_learning_rate: generator learning rate
            dis_learning_rate: discriminator learning rate
            num_samples: number of samples generated in sample_sheet
            start_depth: start training from this depth
            num_workers: number of workers for reading the data
            feedback_factor: number of logs per epoch
            save_dir: directory for saving the models (.bin files)
            checkpoint_factor: save model after these many epochs.
        Returns: None (Writes multiple files to disk)
        """

        print(f"Loaded the dataset with: {len(dataset)} images ...")  # type: ignore
        assert (self.depth - 1) == len(
            batch_sizes
        ), "batch_sizes are not compatible with depth"
        assert (self.depth - 1) == len(epochs), "epochs are not compatible with depth"

        self._toggle_all_networks("train")

        # create the generator and discriminator optimizers
        gen_optim = torch.optim.Adam(
            params=self.gen.parameters(),
            lr=gen_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )
        dis_optim = torch.optim.Adam(
            params=self.dis.parameters(),
            lr=dis_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        # verbose stuff
        print("setting up the image saving mechanism")
        model_dir, log_dir = save_dir / "models", save_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # tensorboard summarywriter:
        summary = SummaryWriter(str(log_dir / "tensorboard"))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth + 1):
            current_res = int(2**current_depth)
            print(f"\n\nCurrently working on Depth: {current_depth}")
            print("Current resolution: %d x %d" % (current_res, current_res))
            depth_list_index = current_depth - 2  # start index from 0
            current_batch_size = batch_sizes[depth_list_index]
            data = get_data_loader(dataset, current_batch_size, num_workers)
            ticker = 1
            
            # because for each depth we can define a different number of epochs
            for epoch in range(1, epochs[depth_list_index] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print(f"\nEpoch: {epoch}")
                total_batches = len(data)
                print(f"Total batches: {total_batches}")

                # compute the fader point
                fader_point = int(
                    (fade_in_percentages[depth_list_index] / 100)
                    * epochs[depth_list_index]
                    * total_batches
                )

                for i, batch in enumerate(data, start=1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    gan_input = batch['A'].to(self.device)
                    images = batch['B'].to(self.device)

                    gen_loss, dis_loss = None, None
                    for _ in range(batch_repeats):
                        # optimize the discriminator:
                        dis_loss = self.optimize_discriminator(
                            loss_fn, dis_optim, gan_input, images, current_depth, alpha
                        )

                        # no idea why this needs to be done after discriminator
                        # iteration, but this is where it is done in the original
                        # code
                        if self.use_ema:
                            update_average(
                                self.gen_shadow, self.gen, beta=self.ema_beta
                            )

                        # optimize the generator:
                        gen_loss = self.optimize_generator(
                            loss_fn, gen_optim, gan_input, images, current_depth, alpha
                        )
                    global_step += 1

                    # provide a loss feedback
                    if (
                        i % max(int(total_batches / max(feedback_factor, 1)), 1) == 0
                        or i == 1
                        or i == total_batches
                    ):
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(
                            "Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                            % (elapsed, i, dis_loss, gen_loss)
                        )
                        summary.add_scalar(
                            "dis_loss", dis_loss, global_step=global_step
                        )
                        summary.add_scalar(
                            "gen_loss", gen_loss, global_step=global_step
                        )

                    # increment the alpha ticker and the step
                    ticker += 1

                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                if (
                    epoch % checkpoint_factor == 0
                    or epoch == 1
                    or epoch == epochs[depth_list_index]
                ):
                    save_file = model_dir / f"depth_{current_depth}_epoch_{epoch}.bin"
                    torch.save(self.get_save_info(gen_optim, dis_optim), save_file)

        self._toggle_all_networks("eval")
        print("Training completed ...")


class CycleGAN:
    def __init__(
        self,
        gen_AB: Generator,
        gen_BA: Generator,
        dis_A: Discriminator,
        dis_B: Discriminator,
        device=torch.device("cuda"),
        use_ema: bool = False,
        ema_beta: float = 0.999,
    ):
        assert gen_AB.depth == gen_BA.depth == dis_A.depth == dis_B.depth, (
            f"Generator and Discriminator depths are not compatible. "
            f"GEN_AB_Depth: {gen_AB.depth}  GEN_BA_Depth: {gen_BA.depth}  DIS_A_Depth: {dis_A.depth}  DIS_B_Depth: {dis_B.depth}"
        )
        self.gen_AB = gen_AB.to(device)
        self.gen_BA = gen_BA.to(device)
        self.dis_A = dis_A.to(device)
        self.dis_B = dis_B.to(device)
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.depth = gen_AB.depth  # any works
        self.device = device

        # if code is to be run on GPU, we can use DataParallel:
        # if device == torch.device("cuda"):
        #     self.gen_AB = DataParallel(self.gen_AB)
        #     self.gen_BA = DataParallel(self.gen_BA)
        #     self.dis_A = DataParallel(self.dis_A)
        #     self.dis_B = DataParallel(self.dis_B)

        print(f"Generator AB Network: {self.gen_AB}")
        print(f"Generator BA Network: {self.gen_BA}")
        print(f"Discriminator A Network: {self.dis_A}")
        print(f"Discriminator B Network: {self.dis_B}")

        # counters to maintain generator and discriminator gradient overflows
        self.gen_overflow_count = 0
        self.dis_overflow_count = 0

    def load_weights_generator(self, saved_model_path: Path) -> Generator:
        # load the data from the saved_model
        loaded_data = torch.load(saved_model_path)
        generator_data = loaded_data['generator']

        generator = Generator(**generator_data["conf"])
        generator.load_state_dict(generator_data["state_dict"])
        generator.to(self.device)

        return generator

    def load_weights_discriminator(self, saved_model_path: Path) -> Discriminator:
        # load the data from the saved_model
        loaded_data = torch.load(saved_model_path)
        discriminator_data = loaded_data['discriminator']

        discriminator = Discriminator(**discriminator_data["conf"])
        discriminator.load_state_dict(discriminator_data["state_dict"])
        discriminator.to(self.device)

        return discriminator

    def optimize_discriminator(
        self,
        loss: CycleGANLoss,
        dis_optim_A: Optimizer,
        dis_optim_B: Optimizer,
        real_A: Tensor,
        real_B: Tensor,
    ) -> float:
        """
        performs a single weight update step on discriminator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            dis_optim_A: discriminator A optimizer
            dis_optim_B: discriminator B optimizer
            real_A: real images from domain A
            real_B: real images from domain B

        Returns: discriminator loss value
        """
        dis_loss = loss.dis_loss(self.gen_AB, self.gen_BA, self.dis_A, self.dis_B, real_A, real_B)

        # optimize discriminator
        dis_optim_A.zero_grad()
        dis_optim_B.zero_grad()
        dis_loss.backward()
        if self._check_grad_ok(self.dis_A) and self._check_grad_ok(self.dis_B):
            dis_optim_A.step()
            dis_optim_B.step()
        else:
            self.dis_overflow_count += 1

        return dis_loss.item()

    def optimize_generator(
        self,
        loss: CycleGANLoss,
        gen_optim_AB: Optimizer,
        gen_optim_BA: Optimizer,
        real_A: Tensor,
        real_B: Tensor,
    ) -> float:
        """
        performs a single weight update step on generator using the batch of data
        and the noise
        Args:
            loss: the loss function to be used for the optimization
            gen_optim_AB: generator AB optimizer
            gen_optim_BA: generator BA optimizer
            real_A: real images from domain A
            real_B: real images from domain B

        Returns: generator loss value
        """

        gen_loss = loss.gen_loss(self.gen_AB, self.gen_BA, self.dis_A, self.dis_B, real_A, real_B)

        # optimize the generator
        gen_optim_AB.zero_grad()
        gen_optim_BA.zero_grad()
        gen_loss.backward()
        if self._check_grad_ok(self.gen_AB) and self._check_grad_ok(self.gen_BA):
            gen_optim_AB.step()
            gen_optim_BA.step()
        else:
            self.gen_overflow_count += 1

        return gen_loss.item()


    def _toggle_all_networks(self, mode="train"):
        for network in (self.gen_AB, self.gen_BA, self.dis_A, self.dis_B):
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

    def get_save_info(self) -> Dict[str, Any]:

        if self.device == torch.device("cpu"):
            generatorAB_save_info = self.gen_AB.get_save_info()
            generatorBA_save_info = self.gen_BA.get_save_info()
            discriminatorA_save_info = self.dis_A.get_save_info()
            discriminatorB_save_info = self.dis_B.get_save_info()
        else:
            generatorAB_save_info = self.gen_AB.module.get_save_info()
            generatorBA_save_info = self.gen_BA.module.get_save_info()
            discriminatorA_save_info = self.dis_A.module.get_save_info()
            discriminatorB_save_info = self.dis_B.module.get_save_info()
        
        save_info = {
            "generatorAB": generatorAB_save_info,
            "generatorBA": generatorBA_save_info,
            "discriminatorA": discriminatorA_save_info,
            "discriminatorB": discriminatorB_save_info,
        }

        return save_info

    def train(
        self,
        dataset: Dataset,
        epochs: int,
        batch_sizes: int,
        loss_fn: CycleGANLoss = CycleGANLoss(),
        batch_repeats: int = 4,
        gen_learning_rate: float = 0.003,
        dis_learning_rate: float = 0.003,
        num_workers: int = 3,
        feedback_factor: int = 100,
        save_dir=Path("./saved_models/cyclegan"),
        checkpoint_factor: int = 10,
        pretrained_model_path: Optional[Path] = None,
        wb_mode: bool = True,
    ):
        print(f"Loaded the dataset with: {len(dataset)} images ...")  # type: ignore

        self._toggle_all_networks("train")

        # create the generator and discriminator optimizers
        gen_optim_AB = torch.optim.Adam(
            params=self.gen_AB.parameters(),
            lr=gen_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )
        
        gen_optim_BA = torch.optim.Adam(
            params=self.gen_BA.parameters(),
            lr=gen_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        dis_optim_A = torch.optim.Adam(
            params=self.dis_A.parameters(),
            lr=dis_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        dis_optim_B = torch.optim.Adam(
            params=self.dis_B.parameters(),
            lr=dis_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        assert pretrained_model_path is not None, "pretrained model path is required"

        self.gen_AB = self.load_weights_generator(pretrained_model_path)
        self.dis_B = self.load_weights_discriminator(pretrained_model_path)

        # verbose stuff
        print("setting up the image saving mechanism")
        model_dir, log_dir = save_dir / "models", save_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # tensorboard summarywriter:
        summary = SummaryWriter(str(log_dir / "tensorboard"))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        print("Starting the training process ... ")
        # because for each depth we can define a different number of epochs
        data = get_data_loader(dataset, batch_sizes, num_workers)
        for epoch in range(1, epochs+1):
            start = timeit.default_timer()  # record time at the start of epoch
            print(f"\nEpoch: {epoch}")
            total_batches = len(data)
            for i, batch in enumerate(data):

                # extract current batch of data for training
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)

                gen_loss, dis_loss = None, None

                dis_loss = self.optimize_discriminator(
                    loss_fn, dis_optim_A, dis_optim_B, real_A, real_B
                )

                gen_loss = self.optimize_generator(
                    loss_fn, gen_optim_AB, gen_optim_BA, real_A, real_B
                )

                global_step += 1

                # provide a loss feedback
                if (
                    i % max(int(total_batches / max(feedback_factor, 1)), 1) == 0
                    or i == 1
                    or i == total_batches
                ):
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(
                        "Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                        % (elapsed, i, dis_loss, gen_loss)
                    )
                    summary.add_scalar(
                        "dis_loss", dis_loss, global_step=global_step
                    )
                    summary.add_scalar(
                        "gen_loss", gen_loss, global_step=global_step
                    )

            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if (
                epoch % checkpoint_factor == 0
                or epoch == 1
            ):
                save_file = model_dir / f"epoch_{epoch}.bin"
                torch.save(self.get_save_info(), save_file)

        self._toggle_all_networks("eval")
        print("Training completed ...")
