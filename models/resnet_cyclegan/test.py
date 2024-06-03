import torch
import argparse
import numpy as np
from torch import Tensor
from data import RGBTileDataset
from model import GeneratorResNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import show_xyz, show_rgb

parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
parser.add_argument("--dataset_name", default="melbourne")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--nepochs", type=int, default=5, help="saved model epochs")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# load model
model_path = "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.nepochs)
model_g = GeneratorResNet(input_shape, opt.n_residual_blocks)
if cuda:
    model_g.cuda()
model_g.load_state_dict(torch.load(model_path))

# print all weights
for name, param in model_g.named_parameters():
    print(name, param)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

test_set = RGBTileDataset(dataset=opt.dataset_name, image_set="test")
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)
for i, batch in enumerate(test_dl):
    
    # Model inputs
    xyz_input = Variable(batch["A"].type(Tensor))
    # rgb_label = Variable(batch["B"].type(Tensor))
    out_imgs = model_g(xyz_input)
    label = out_imgs.cpu().data.numpy()
    show_xyz(batch["A"].numpy(), cols=2)
    # show_rgb(batch["B"].numpy(), cols=2)
    show_rgb(label, cols=2)
