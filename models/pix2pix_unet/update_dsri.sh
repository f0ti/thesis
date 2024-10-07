#!/usr/bin/sh

oc cp train.py jupyterlab-gpu-2-6vkn9:/root/thesis/models/pix2pix_unet/train.py
oc cp test.py jupyterlab-gpu-2-6vkn9:/root/thesis/models/pix2pix_unet/test.py
oc cp data.py jupyterlab-gpu-2-6vkn9:/root/thesis/models/pix2pix_unet/data.py
oc cp model.py jupyterlab-gpu-2-6vkn9:/root/thesis/models/pix2pix_unet/model.py
oc cp utils.py jupyterlab-gpu-2-6vkn9:/root/thesis/models/pix2pix_unet/utils.py
