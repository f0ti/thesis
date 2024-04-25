from os.path import join
from dataset import TileDataset


def get_training_set(dataset):
    return TileDataset(dataset=dataset, image_set="train")

def get_test_set(dataset):
    return TileDataset(dataset=dataset, image_set="test")
