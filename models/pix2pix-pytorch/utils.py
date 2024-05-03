import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def read_config_from_file(file_path="config.py"):
    cfg = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            key, value = key.strip().lower(), value.strip()
            cfg[key] = value
    
    return cfg
