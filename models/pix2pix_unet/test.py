import os
import torch
import argparse
import numpy as np
from cleanfid import fid
from dotenv import load_dotenv
from data import RGBTileDataset
from model import GeneratorUNet
from torch.utils.data import DataLoader
from utils import show_xyz, show_rgb, show_diff

load_dotenv()

parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
parser.add_argument("--dataset", default="melbourne-top")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--nepochs", type=int, default=10, help="saved model epochs")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# load model
model_path = "models_vault/{}/generator_{}.pth".format(opt.dataset, opt.nepochs)
model_g = GeneratorUNet().to(device)
model_g.load_state_dict(torch.load(model_path))

test_set = RGBTileDataset(dataset=opt.dataset, image_set="test")
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

image_writing_path = "generated_images"
if not os.path.exists(image_writing_path):
    os.makedirs(image_writing_path)

for i, batch in enumerate(test_dl, start=1):
    # # Model inputs
    xyz_input = batch["A"].cuda()
    out_imgs = model_g(xyz_input)
    label = out_imgs.cpu().data

    # for img_num in range(opt.batch_size):
    #     print(f"Saving image number {i}_{img_num} out of {len(test_dl)}")
    #     np.save(f"{image_writing_path}/gen_imgs_{i}_{img_num}.npy", label.numpy().astype(np.float32)[img_num])  # typing: ignore

    # get Visual Saliency-induced Index (VSI)
    # vsi_score = vsi(label.type(Tensor), batch["B"].type(Tensor))
    # print(vsi_score)

    # XYZ (only Z)
    # show_xyz(batch["A"].numpy(), cols=2)
    
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