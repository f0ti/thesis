import os
import json
import urllib
import zipfile
from collections import defaultdict
from typing import Iterator, Optional
from tqdm import tqdm


DATASET_URL = {
    'melbourne': '',
}

def download_and_extract_archive(
    url: str,
    download_root: str,
) -> None:
    root = os.path.expanduser(download_root)
    filename = os.path.basename(url)

    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:
        raise e

    # extract the archive

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {download_root}")

    with zipfile.ZipFile(archive, "r", compression=zipfile.ZIP_STORED) as zip:
        zip.extractall(download_root)


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "github.com/f0ti"})
    ) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


def generate_all_classes() -> None:
    dataset_path = "data/tiny-imagenet-200"
    words_path = os.path.join(dataset_path, "words.txt")

    wnid_to_word = []
    with open(words_path, "r") as wp:
        for line in wp:
            wnid, str_labels = line.split("\t")
            wnid_to_word.append({"wnid": wnid, "labels": str_labels.strip()})

    with open("classes.json", "w") as outfile:
        json.dump(wnid_to_word, outfile)


def get_class_label(class_index: int) -> str:
    return CLASS_LABELS[class_index]["labels"]