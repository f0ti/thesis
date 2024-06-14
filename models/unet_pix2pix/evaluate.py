import torch
from torch.utils.data import DataLoader
from piq import FID

from data import RGBDomainIndependentDataset

def evaluate_fid(xyz_dl, rgb_dl):
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(xyz_dl)
    second_feats = fid_metric.compute_feats(rgb_dl)
    fid: torch.Tensor = fid_metric(first_feats, second_feats)
    return fid.item()

def main():
    xyz_data = RGBDomainIndependentDataset(dataset="melbourne", domain="xyz", image_set="test")
    rgb_data = RGBDomainIndependentDataset(dataset="melbourne", domain="rgb", image_set="test")
    xyz_dl = DataLoader(dataset=xyz_data, batch_size=1)
    rgb_dl = DataLoader(dataset=rgb_data, batch_size=1)

    print(next(iter(rgb_dl))['images'].min(), next(iter(rgb_dl))['images'].max())
    print(next(iter(xyz_dl))['images'].min(), next(iter(xyz_dl))['images'].max())

    fid = evaluate_fid(xyz_dl, rgb_dl)
    print(f"FID: {fid}")

if __name__ == "__main__":
    main()
