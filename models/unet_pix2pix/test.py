import torch
import argparse
from data import RGBTileDataset
from model import GeneratorUNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import show_xyz, show_rgb, show_diff

parser = argparse.ArgumentParser(description="pix2pix-pytorch-implementation")
parser.add_argument("--dataset", default="melbourne")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--nepochs", type=int, default=10, help="saved model epochs")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

# load model
model_path = "models_vault/{}/generator_{}.pth".format(opt.dataset, opt.nepochs)
model_g = GeneratorUNet()
if cuda:
    model_g.cuda()
model_g.load_state_dict(torch.load(model_path))

# print all weights
# for name, param in model_g.named_parameters():
#     print(name, param)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

test_set = RGBTileDataset(dataset=opt.dataset, image_set="test")
test_dl = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

for i, batch in enumerate(test_dl):
    
    # Model inputs
    xyz_input = Variable(batch["A"].type(Tensor))
    out_imgs = model_g(xyz_input)
    label = out_imgs.cpu().data.numpy()
    
    # XYZ (only Z)
    show_xyz(batch["A"].numpy(), cols=2)
    
    # ground truth RGB
    show_rgb(batch["B"].numpy(), cols=2)

    # predicted RGB
    show_rgb(label, cols=2)

    print(batch["B"].numpy().dtype, label.dtype)
    print(batch["B"].numpy().shape, label.shape)

    # difference
    show_diff(batch["B"].numpy(), label, cols=2)

