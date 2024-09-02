import os
import sys

def filter_from_blacklist(blacklist_name: str = "blacklist_0.97.txt", dataset_name: str = "melbourne-top"):
    with open(blacklist_name, "r") as f:
        blacklist = f.readlines()
        blacklist = [line.strip() for line in blacklist]

    for line in blacklist:
        os.remove(f"../{dataset_name}/rgb_data/{line}")
        os.remove(f"../{dataset_name}/xyz_data/{line}")

if __name__ == "__main__":
    blacklist_name = sys.argv[1]
    dataset_name = sys.argv[2]
    filter_from_blacklist(blacklist_name)
