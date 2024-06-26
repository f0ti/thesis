import argparse
import os
import numpy as np
import time
import datetime
import sys
import wandb
import torch
import shortuuid

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data import *
from loss import *
from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=11, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="melbourne", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=0, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
parser.add_argument("--wb", type=int, default=1, help="weights and biases")

opt = parser.parse_args()

if opt.wb:
    wandb.init(project="thesis", config=vars(opt))

os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
if opt.sample_interval:
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# criterion_vifp = VIFLoss()
criterion_fid = FrechetID()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

print("Initializing models")
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    # criterion_vifp.cuda()
    criterion_fid.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("models_vault/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("models_vault/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

print('Loading datasets...')
train_set = RGBTileDataset(dataset=opt.dataset_name, image_set="train")
test_set = RGBTileDataset(dataset=opt.dataset_name, image_set="test")
train_dl = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(test_dl))
    real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dl):
        
        # show_xyz(batch["A"].numpy(), cols=4)
        # show_rgb(batch["B"].numpy(), cols=4)
        
        # Model inputs
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Visual Information Fidelity loss
        # loss_vifp = criterion_vifp(fake_B, real_B)
        # Frechet Inception Distance loss
        loss_fid = criterion_fid(fake_B, real_B)

        loss_G = loss_GAN + loss_pixel * lambda_pixel + loss_fid

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dl) + i
        batches_left = opt.n_epochs * len(train_dl) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if opt.wb:
            wandb.log({"Loss_D": loss_D.item(), "Loss_G": loss_G.item()})

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dl),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if opt.sample_interval:
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        if opt.wb:
            saving_dir = os.path.join("saved_models", wandb.run.name)
            os.makedirs(saving_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(saving_dir, f"generator_{epoch}.pth"))
        else:
            from random import randint
            saving_dir = os.path.join("saved_models", str(randint(1, 1000)))
            torch.save(generator.state_dict(), os.path.join(saving_dir, f"generator_{epoch}.pth"))
