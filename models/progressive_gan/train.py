""" script for training a ProGAN (Progressively grown gan model) """

import argparse
from pathlib import Path

import torch
from torch.backends import cudnn

from data import RGBTileDataset, get_transform
from gan import CycleGAN, ProGAN
from models.progressive_gan.losses import CycleGANLoss
from networks import Discriminator, Generator, load_models
from utils import str2bool, str2GANLoss

# turn fast mode on
cudnn.benchmark = True

# define the device for the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments() -> argparse.Namespace:
    """
    command line arguments parser
    Returns: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        "Train Progressively grown GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments (input path to the data and the output directory for saving training assets)
    parser.add_argument("--output_dir", action="store", type=Path, default=Path("./saved_models"), required=False,
                        help="Path to the directory for saving the logs and models")
    
    # dataset related options:
    parser.add_argument("--rec_dir", action="store", type=str2bool, default=True, required=False,
                        help="whether images are stored under one folder or under a recursive dir structure")
    parser.add_argument("--flip_horizontal", action="store", type=str2bool, default=False, required=False,
                        help="whether to apply mirror (horizontal) augmentation")

    # model architecture related options:
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--depth", action="store", type=int, default=8, required=False,
                        help="depth of the generator and the discriminator. Starts from 2. "
                             "Example 2 --> (4x4) | 3 --> (8x8) ... | 10 --> (1024x1024)")
    parser.add_argument("--num_channels", action="store", type=int, default=3, required=False,
                        help="number of channels in the image data")
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--latent_size", action="store", type=int, default=256, required=False,
                        help="latent size of the generator and the discriminator")

    # training related options:
    parser.add_argument("--use_eql", action="store", type=str2bool, default=True, required=False,
                        help="whether to use the equalized learning rate")
    parser.add_argument("--use_ema", action="store", type=str2bool, default=True, required=False,
                        help="whether to use the exponential moving average of generator weights. "
                             "Keeps two copies of the generator model; an instantaneous one and "
                             "the averaged one.")
    parser.add_argument("--ema_beta", action="store", type=float, default=0.999, required=False,
                        help="value of the ema beta")
    
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--epochs", action="store", type=int, required=False, nargs="+",
                        default=[5 for _ in range(7)],  # because there are 7 stages
                        help="number of epochs over the training dataset per stage")
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--batch_sizes", action="store", type=int, required=False, nargs="+",
                        default=[8, 8, 8, 4, 4, 2, 1],  # (4x4), (8x8), (16x16), (32x32), (64x64), (128x128), (256x256)
                        help="batch size used for training the model per stage")
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--batch_repeats", action="store", type=int, required=False, default=2,
                        help="number of G and D steps executed per training iteration")
    parser.add_argument("--fade_in_percentages", action="store", type=int, required=False, nargs="+",
                        default=[50 for _ in range(7)],
                        help="number of iterations for which fading of new layer happens. Measured in percentage")
    parser.add_argument("--loss_fn", action="store", type=str2GANLoss, required=False, default="wgan_gp",
                        help="loss function used for training the GAN. "
                             "Current options: [wgan_gp, standard_gan]")
    parser.add_argument("--g_lrate", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the generator")
    parser.add_argument("--d_lrate", action="store", type=float, required=False, default=0.003,
                        help="learning rate used by the discriminator")
    parser.add_argument("--num_feedback_samples", action="store", type=int, required=False, default=4,
                        help="number of samples used for fixed seed gan feedback")
    parser.add_argument("--start_depth", action="store", type=int, required=False, default=2,
                        help="resolution to start the training from. "
                             "Example 2 --> (4x4) | 3 --> (8x8) ... | 10 --> (1024x1024). "
                             "Note that this is not a way to restart a partial training. "
                             "Resuming is not supported currently. But will soon be.")
    parser.add_argument("--num_workers", action="store", type=int, required=False, default=8,
                        help="number of dataloader subprocesses. It's a pytorch thing, you can ignore it ;)."
                             " Leave it to the default value unless things are weirdly slow for you.")
    parser.add_argument("--feedback_factor", action="store", type=int, required=False, default=10,
                        help="number of feedback logs written per epoch")
    # ************** IMPORTANT HYPERPARAMETER
    parser.add_argument("--checkpoint_factor", action="store", type=int, required=False, default=5,
                        help="number of epochs after which a model snapshot is saved per training stage")
    parser.add_argument("--wb_mode", action="store", type=bool, required=False, default=False, help="weights and biases mode")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def train_progan(args: argparse.Namespace) -> None:
    """
    method to train the progan (progressively grown gan) given the configuration parameters
    Args:
        args: configuration used for the training
    Returns: None
    """
    print(f"Selected arguments: {args}")

    generator = Generator(
        depth=args.depth,
        num_channels=args.num_channels,
        use_eql=args.use_eql,
    )
    discriminator = Discriminator(
        depth=args.depth,
        num_channels=args.num_channels,
        latent_size=args.latent_size,
        use_eql=args.use_eql,
    )

    progan = ProGAN(
        generator,
        discriminator,
        device=device,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
    )

    progan.train(
        dataset=RGBTileDataset(
            image_set="train",
            transform=get_transform(
                new_size=(int(2**args.depth), int(2**args.depth)),
                flip_horizontal=args.flip_horizontal,
            ),
        ),
        epochs=args.epochs,
        batch_sizes=args.batch_sizes,
        fade_in_percentages=args.fade_in_percentages,
        loss_fn=args.loss_fn,
        batch_repeats=args.batch_repeats,
        gen_learning_rate=args.g_lrate,
        dis_learning_rate=args.d_lrate,
        num_samples=args.num_feedback_samples,
        start_depth=args.start_depth,
        num_workers=args.num_workers,
        feedback_factor=args.feedback_factor,
        checkpoint_factor=args.checkpoint_factor,
        save_dir=args.output_dir,
        wb_mode=args.wb_mode,
    )


def train_cyclegan(args: argparse.Namespace) -> None:
    """
    method to train the cyclegan given the configuration parameters
    Args:
        args: configuration used for the training
    Returns: None
    """
    print(f"Selected arguments: {args}")

    generator_AB = Generator(
        depth=args.depth
    )

    generator_BA = Generator(
        depth=args.depth,
    )

    discriminator_A = Discriminator(
        depth=args.depth,
    )

    discriminator_B = Discriminator(
        depth=args.depth,
    )

    cyclegan = CycleGAN(
        generator_AB,
        generator_BA,
        discriminator_A,
        discriminator_B,
        device=device,
        use_ema=args.use_ema,
        ema_beta=args.ema_beta,
    )

    cyclegan.train(
        dataset=RGBTileDataset(
            image_set="train",
            transform=get_transform(
                new_size=(int(2**args.depth), int(2**args.depth)),
                flip_horizontal=args.flip_horizontal,
            ),
        ),
        epochs=10,
        batch_sizes=1,
        loss_fn=CycleGANLoss(),
        gen_learning_rate=args.g_lrate,
        dis_learning_rate=args.d_lrate,
        num_workers=args.num_workers,
        feedback_factor=args.feedback_factor,
        checkpoint_factor=args.checkpoint_factor,
        save_dir=args.output_dir,
        wb_mode=args.wb_mode,
    )


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    train_progan(parse_arguments())


if __name__ == "__main__":
    main()
