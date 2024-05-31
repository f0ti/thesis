from data import *
from torch.utils.data import DataLoader

print('Loading datasets...')
train_set = RGBTileDataset(dataset="melbourne", image_set="train")
train_dl = DataLoader(dataset=train_set, num_workers=8, batch_size=4, shuffle=True)

for x in train_dl:
    print(x["A"].shape, x["B"].shape)
    break