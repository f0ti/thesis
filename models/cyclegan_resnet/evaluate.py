""" script for computing the fid of a trained model when compared with the dataset images """
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm

import os
import torch
import numpy as np
from cleanfid import fid
from torch.backends import cudnn
from model import GeneratorResNet

# turn fast mode on
cudnn.benchmark = True

def create_generator_from_saved_model(run_name, nepochs):
    model_path = "saved_models/%s/G_AB_%d.pth" % (run_name, nepochs)
    model_g = GeneratorResNet((3, 256, 256), 9)
    
    model_g.load_state_dict(torch.load(model_path))

    return model_g


def post_process_generated_images(gen_imgs):
    return (gen_imgs * 255.0).detach().cpu().numpy().astype(np.uint8)


def parse_arguments() -> argparse.Namespace:
    """
    Returns: parsed arguments object
    """
    parser = argparse.ArgumentParser("ProGAN fid_score computation tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    # fmt: off
    parser.add_argument("--run_name", action="store", type=str, default="mild-fire-52",
                        help="run name of the model")
    parser.add_argument("--nepochs", action="store", type=int, default=10, required=False,)
    parser.add_argument("--generated_images_path", action="store", type=Path, default=None, required=False,
                        help="path to the directory where the generated images are to be written. "
                             "Uses a temporary directory by default. Provide this path if you'd like "
                             "to see the generated images yourself :).")
    parser.add_argument("--batch_size", action="store", type=int, default=4, required=False,
                        help="batch size used for generating random images")
    parser.add_argument("--num_generated_images", action="store", type=int, default=50_000, required=False,
                        help="number of generated images used for computing the FID")
    # fmt: on

    args = parser.parse_args()

    return args


def compute_fid(args: argparse.Namespace) -> None:
    """
    compute the fid for a given trained pro-gan model
    Args:
        args: configuration used for the fid computation
    Returns: None

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data from the trained-model
    print(f"loading data from the trained model at: {args.run_name}")
    generator = create_generator_from_saved_model(args.run_name, args.nepochs).to(device)

    # create the generated images directory:
    if args.generated_images_path is not None:
        args.generated_images_path.mkdir(parents=True, exist_ok=True)
    generated_images_path = (
        args.generated_images_path
        if args.generated_images_path is not None
        else tempfile.TemporaryDirectory()
    )
    if args.generated_images_path is None:
        image_writing_path = Path(generated_images_path.name)
    else:
        image_writing_path = generated_images_path

    print("generating random images from the trained generator ...")
    with torch.no_grad():
        for img_num in tqdm(range(0, args.num_generated_images, args.batch_size)):
            num_imgs = min(args.batch_size, args.num_generated_images - img_num)
            random_latents = torch.randn(num_imgs, generator.latent_size, device=device)
            gen_imgs = generator(random_latents)

            np.save(image_writing_path / f"gen_imgs_{img_num}.npy", gen_imgs.cpu().numpy().astype(np.float32))  # typing: ignore

    # compute the fid once all images are generated
    print("computing fid ...")
    score = fid.compute_fid(
        fdir1=args.dataset_path,
        fdir2=image_writing_path,
        mode="clean",
        num_workers=4,
    )
    print(f"fid score: {score: .3f}")

    # most importantly, don't forget to do the cleanup on the temporary directory:
    if hasattr(generated_images_path, "cleanup"):
        generated_images_path.cleanup()


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    compute_fid(parse_arguments())


if __name__ == "__main__":
    main()