import os
import sys

def filter_from_blacklist(blacklist_name: str, dataset_name: str):
    with open(blacklist_name, "r") as f:
        blacklist = f.readlines()
        blacklist = [line.strip() for line in blacklist]

    print("Removing images from blacklist: ", len(blacklist))
    for line in blacklist:
        try:
            if dataset_name == "melbourne-top":
                os.remove(f"../{dataset_name}/xyz_data/{line}")
            elif dataset_name == "melbourne-z-top":
                os.remove(f"../{dataset_name}/z_data/{line}")
            else:
                print(f"Dataset {dataset_name} not supported for XYZRGB")
            os.remove(f"../{dataset_name}/rgb_data/{line}")
        except FileNotFoundError:
            print(f"File not found: {line}")

if __name__ == "__main__":
    blacklist_name = sys.argv[1]
    dataset_name = sys.argv[2]
    filter_from_blacklist(blacklist_name, dataset_name)
