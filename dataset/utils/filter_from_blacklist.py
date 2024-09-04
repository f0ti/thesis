import os
import sys

def filter_from_blacklist(blacklist_name: str, dataset_name: str):
    with open(blacklist_name, "r") as f:
        blacklist = f.readlines()
        blacklist = [line.strip() for line in blacklist]

    print("Removing images from blacklist: ", len(blacklist))
    for line in blacklist:
        try:
            os.remove(f"../{dataset_name}/rgb_data/{line}")
            os.remove(f"../{dataset_name}/xyz_data/{line}")
        except FileNotFoundError:
            print(f"File not found: {line}")

if __name__ == "__main__":
    blacklist_name = sys.argv[1]
    dataset_name = sys.argv[2]
    filter_from_blacklist(blacklist_name, dataset_name)
