""" Module implementing ProGAN which is trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import copy
import datetime
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
from losses import CycleGANLoss, GANLoss, WganGP, TotalVariationLoss, SSIMLoss, SDILoss
from networks import Discriminator, Generator
from utils import adjust_dynamic_range

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
        
        generatorAB_save_info = self.gen_AB.get_save_info()
        generatorBA_save_info = self.gen_BA.get_save_info()
        discriminatorA_save_info = self.dis_A.get_save_info()
        discriminatorB_save_info = self.dis_B.get_save_info()

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
    ):
        print(f"Loaded the dataset with: {len(dataset)} images ...")  # type: ignore

        self._toggle_all_networks("train")

        # create the generator and discriminator optimizersconda install python=3.9

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

        print("Loading the pretrained model ...")
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
        data = get_data_loader(dataset, batch_sizes, num_workers)
        for epoch in range(1, epochs+1):
            start = timeit.default_timer()  # record time at the start of epoch
            print(f"\nEpoch: {epoch}")
            total_batches = len(data)
            for i, batch in enumerate(data, start=1):

                # extract current batch of data for training
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)

                gen_loss, dis_loss = 0, 0  # because None breaks the print

                # as proposed in the original paper of CycleGAN, we need to implement
                # a buffer of n samples to store the generated samples, as presented
                # in the paper https://arxiv.org/pdf/1612.07828.pdf. To optimize the
                # discriminator, b/2 samples are drawn from the buffer and b/2 samples
                # are drawn from the current minibatch. Since I am using a batch size
                # of 1, this does not make sense, so will skip this part for now.
                
                if i % 10 == 0:
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
