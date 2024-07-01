import torch
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from piq import vsi

from utils import show_xyz, show_rgb, show_diff
from data import RGBTileDataset
from model import GeneratorResNet

parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
parser.add_argument("--dataset_name", default="melbourne-top")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--nepochs", type=int, default=0, help="saved model epochs")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--run_name", default="silver-meadow-37", help="run name")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# load model
model_path = "saved_models/%s/G_AB_%d.pth" % (opt.run_name, opt.nepochs)
# model_path = "models_vault/G_AB_10.pth"
model_g = GeneratorResNet(input_shape, opt.n_residual_blocks)
if cuda:
    model_g.cuda()
model_g.load_state_dict(torch.load(model_path))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

test_set = RGBTileDataset(dataset=opt.dataset_name, image_set="test")
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

# print parameters
for p in model_g.named_parameters():
    print(p)

for i, batch in enumerate(test_dl):
    # Model inputs
    xyz_input = Variable(batch["A"].type(Tensor))
    out_imgs = model_g(xyz_input)
    label = out_imgs.cpu().data
    # label = label / 2.0

    # get Visual Saliency-induced Index (VSI)
    # vsi_score = vsi(label.type(Tensor), batch["B"].type(Tensor))
    # print(vsi_score)

    print(label)
    
    # XYZ (only Z)
    show_xyz(batch["A"].numpy(), cols=2)
    
    # ground truth RGB
    show_rgb(batch["B"].numpy(), cols=2)

    # predicted RGB
    show_rgb(label.numpy(), cols=2)

    # difference
    show_diff(batch["B"].numpy(), label.numpy(), cols=2)
