import os
from tracemalloc import start
import torch
import argparse
import numpy as np
from cleanfid import fid
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from utils import show_xyz, show_rgb, show_z, show_diff, adjust_dynamic_range
from data import MelbourneZRGB
from model import GeneratorResNet

load_dotenv()

parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
parser.add_argument("--dataset_name", default="melbourne-z-top")
parser.add_argument("--model_path", default="G_AB_8.pth")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--nepochs", type=int, default=10, help="saved model epochs")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--run_name", default="mild-fire-52", help="run name")
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model
model_g = GeneratorResNet(1, 3).to(device)
model_g.load_state_dict(torch.load(opt.model_path))

# load data
test_set = MelbourneZRGB(dataset=opt.dataset_name, image_set="test")
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

# save data
image_writing_path = "test_generated_images"
if not os.path.exists(image_writing_path):
    os.makedirs(image_writing_path)

for i, batch in enumerate(test_dl):
    # # Model inputs
    xyz_input = batch["A"].to(device)
    out_imgs = model_g(xyz_input)
    label = out_imgs.cpu().data

    # for img_num in range(opt.batch_size):
    #     print(f"Saving image number {i}_{img_num} out of {len(test_dl)}")
    #     np.save(f"{image_writing_path}/gen_imgs_{i}_{img_num}.npy", label.numpy().astype(np.float32)[img_num])  # typing: ignore

    # XYZ (only Z)
    # show_z(batch["A"].numpy(), cols=2)
    
    # ground truth RGB
    show_rgb(batch["B"].numpy(), cols=2)

    # predicted RGB
    show_rgb(label.numpy(), cols=2)

    # difference
    # show_diff(batch["B"].numpy(), label.numpy(), cols=2)

# data_base_dir = os.environ.get("DATASET_ROOT") or "."
# dataset_path = os.path.join(data_base_dir, "melbourne-top")
# rgb_path = os.path.join(dataset_path, "test", "rgb_data")

# print("computing fid ...")
# score = fid.compute_fid(
#     fdir1=rgb_path,
#     fdir2=image_writing_path,
#     mode="clean",
#     num_workers=4,
# )

# print(score)