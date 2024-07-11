import argparse
import os
import numpy as np
import itertools
import datetime
import time
import wandb
import torch

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from utils import *
from model import *
from loss import *
from data import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=7, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="melbourne-top", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--threads", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=0, help="interval between saving generator outputs")
parser.add_argument("--ckpt", type=int, default=2, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=8.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_tv", type=float, default=0.1, help="total variation loss weight")
parser.add_argument("--lambda_ssim", type=float, default=1.0, help="structural similarity loss weight")
parser.add_argument("--lambda_sdi", type=float, default=1.0, help="structural distortion index loss weight")
parser.add_argument("--wb", type=int, default=1, help="weights and biases")
opt = parser.parse_args()

if opt.wb:
    wandb.init(project="thesis", config=vars(opt))

# Create sample and checkpoint directories
if opt.sample_interval:
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_tv = TotalVariationLoss()
criterion_ssim = SSIMLoss()
criterion_sdi = SDILoss()

cuda = torch.cuda.is_available()

input_shape = (IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    criterion_tv.cuda()
    criterion_ssim.cuda()
    criterion_sdi.cuda()

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step
)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

print('Loading datasets...')
train_set = RGBTileDataset(dataset=opt.dataset_name, image_set="train")
test_set = RGBTileDataset(dataset=opt.dataset_name, image_set="test")
train_dl = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_dl))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs["A"].cuda()
    fake_B = G_AB(real_A)
    real_B = imgs["B"].cuda()
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epochs):
    for i, batch in enumerate(train_dl):

        # show_xyz(batch["A"].numpy(), cols=4)
        # show_rgb(batch["B"].numpy(), cols=4)

        # Set model input
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), *D_A.output_shape)).cuda()
        fake = torch.zeros((real_A.size(0), *D_A.output_shape)).cuda()

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

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
        # loss_tv = criterion_tv(fake_A)  # do it onlym for the forward pass

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

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        # TODO: Freeze the discriminator weights 1/10

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

    if opt.ckpt != -1 and epoch != 0 and epoch % opt.ckpt == 0:
        # Save model checkpoints
        print("\nSaving models...")
        if opt.wb and wandb.run is not None:
            saving_dir = os.path.join("saved_models", wandb.run.name)
            os.makedirs(saving_dir, exist_ok=True)
            torch.save(G_AB.state_dict(), os.path.join(saving_dir, f"G_AB_{epoch}.pth"))
            torch.save(G_BA.state_dict(), os.path.join(saving_dir, f"G_BA_{epoch}.pth"))
        else:
            from random import randint
            saving_dir = os.path.join("saved_models", str(randint(1, 1000)))
            torch.save(G_AB.state_dict(), os.path.join(saving_dir, f"G_AB_{epoch}.pth"))
            torch.save(G_BA.state_dict(), os.path.join(saving_dir, f"G_AB_{epoch}.pth"))