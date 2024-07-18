import argparse
from pathlib import Path

import torch
from torch.backends import cudnn

from networks import create_generator_from_saved_model
from utils import post_process_coordinate_images, post_process_generated_images, show_diff, show_rgb, show_xyz
from data import RGBTileDataset, get_data_loader

# turn fast mode on
cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    """
    Returns: parsed arguments object
    """
    parser = argparse.ArgumentParser("ProGAN fid_score computation tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    # fmt: off
    # required arguments
    parser.add_argument("model_path", action="store", type=Path,
                        help="path to the trained_model.bin file")
    parser.add_argument("dataset_name", action="store", type=str,
                        help="dataset name, one of ['melbourne-top', 'melbourne-side', 'melbourne-all']")

    # optional arguments
    parser.add_argument("--generated_images_path", action="store", type=Path, default=None, required=False,
                        help="path to the directory where the generated images are to be written. "
                             "Uses a temporary directory by default. Provide this path if you'd like "
                             "to see the generated images yourself :).")
    parser.add_argument("--batch_size", action="store", type=int, default=4, required=False,
                        help="batch size used for generating random images")
    # fmt: on

    args = parser.parse_args()

    return args


def show_generated_images(args: argparse.Namespace) -> None:
    """
    show the generated images from the trained model, together
    with the ground truth images from the dataset
    Args:
        args: configuration used for the fid computation
    Returns: None

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data from the trained-model
    print(f"loading data from the trained model at: {args.model_path}")
    generator = create_generator_from_saved_model(args.model_path).to(device)

    test_set = RGBTileDataset(dataset=args.dataset_name, image_set="test")
    test_dl = get_data_loader(test_set, batch_size=args.batch_size)

    print("generating random images from the trained generator ...")
    with torch.no_grad():
        for batch in test_dl:
            gen_imgs = post_process_generated_images(generator(batch["A"].to(device)))
            
            xyz_imgs = post_process_coordinate_images(batch["A"].to(device))
            grt_imgs = post_process_generated_images(batch["B"].to(device))

            # XYZ (only Z)
            show_xyz(xyz_imgs, cols=2)
            
            # ground truth RGB
            show_rgb(grt_imgs, cols=2)

            # predicted RGB
            show_rgb(gen_imgs, cols=2)

            # difference
            show_diff(grt_imgs, gen_imgs, cols=2)


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    show_generated_images(parse_arguments())


if __name__ == "__main__":
    main()