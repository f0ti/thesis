import os
import time
import sys
import torch
import argparse
import datetime

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

from data import *
from loss import *
from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="estonia-z", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")

opt = parser.parse_args()

date_now = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
name = f"{date_now}_{opt.dataset_name}"
image_dir = "sampled_images/%s" % name
model_dir = "saved_models/%s" % name

os.makedirs(image_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

print("Initializing models")
generator = GeneratorUNet(in_channels=1, out_channels=3)
discriminator = Discriminator(in_channels=4)

generator = generator.cuda()
discriminator = discriminator.cuda()
criterion_GAN.cuda()
criterion_pixelwise.cuda()

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
if opt.dataset_name == "melbourne-top":
    train_set = RGBTileDataset(dataset=opt.dataset_name, image_set="train")
    test_set = RGBTileDataset(dataset=opt.dataset_name, image_set="test")
    train_dl = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)
elif opt.dataset_name == "estonia-z":
    train_set = EstoniaZRGB(dataset=opt.dataset_name, image_set="train")
    test_set = EstoniaZRGB(dataset=opt.dataset_name, image_set="test")
    sample_set = EstoniaZRGB(dataset=opt.dataset_name, image_set="test", max_samples=9)
    train_dl = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)
    sample_dl = DataLoader(dataset=sample_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dl):
        real_A = batch["A"].cuda()
        real_B = batch["B"].cuda()

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), *patch)).cuda()
        fake = torch.zeros((real_A.size(0), *patch)).cuda()

        # Generator train
        optimizer_G.zero_grad()
        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + loss_pixel * lambda_pixel

        loss_G.backward()
        optimizer_G.step()

        # Discriminator train
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

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dl),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
            )
        )

        # If at sample interval save image
        if opt.sample_interval and i % opt.sample_interval == 0:
            real_images, fake_images = Tensor([]).cuda(), Tensor([]).cuda()
            for batch in sample_dl:
                real_A = batch["A"].to("cuda")
                fake_B = generator(real_A)
                if adjust_range:
                    fake_B = adjust_range(fake_B, (-1, 1), (0, 1))
                fake_images = torch.cat((fake_images, fake_B), 0)
                if epoch == 0:
                    real_images = torch.cat((real_images, batch["B"].cuda()), 0)

            fake_grid = make_grid(fake_images, nrow=3, normalize=True)
            if epoch == 0:
                real_grid = make_grid(real_images, nrow=3, normalize=True)
                save_image(real_grid, os.path.join(image_dir, f"real_B_{str(epoch).zfill(4)}_{i}.png"))
            save_image(fake_grid, os.path.join(image_dir, f"fake_B_{str(epoch).zfill(4)}_{i}.png"))

            generator.train()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_{epoch}.pth"))
