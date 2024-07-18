import argparse
from pathlib import Path

import torch
from torch.backends import cudnn

from networks import create_generator_from_saved_model
from utils import post_process_generated_images, post_process_coordinate_images, show_diff, show_rgb, show_xyz
from data import RGBTileDataset, get_data_loader

# turn fast mode on
cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    """
    Returns: parsed arguments object
    """
    parser = argparse.ArgumentParser("ProGAN inference tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    # fmt: off
    # required arguments
    parser.add_argument("model_path", action="store", type=Path,
                        help="path to the trained_model.bin file")
    parser.add_argument("dataset_name", action="store", type=str, default="melbourne-top",
                        help="dataset name, one of ['melbourne-top', 'melbourne-side', 'melbourne-all']")
    parser.add_argument("--batch_size", action="store", type=int, default=4, required=False,
                        help="batch size used for generating random images")
    # fmt: on

    args = parser.parse_args()

    return args


def inference(args: argparse.Namespace) -> None:
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
            
            xyz_imgs = post_process_coordinate_images(batch["A"])
            grt_imgs = post_process_generated_images(batch["B"])

            # XYZ (only Z)
            show_xyz(xyz_imgs, cols=args.batch_size)
            
            # ground truth RGB
            show_rgb(grt_imgs, cols=args.batch_size)

            # predicted RGB
            show_rgb(gen_imgs, cols=args.batch_size)

            # difference
            show_diff(grt_imgs, gen_imgs, cols=args.batch_size)


def main() -> None:
    """
    Main function of the script
    Returns: None
    """
    inference(parse_arguments())


if __name__ == "__main__":
    main()
