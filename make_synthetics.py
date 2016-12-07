import os
import gzip
import glob
import json
import pickle
import random
import numpy as np

from skimage.io import imread
from skimage import transform

from PIL import Image

path = os.path.join('data', 'synthetics', 'cropped', '*.jpg')
synth_files = glob.glob(path)
print synth_files

path = os.path.join('data', 'train_raw', 'train', 'NoF', '*.jpg')
bkg_files = glob.glob(path)
print bkg_files

for i in range(1000):
    fish_idx = random.randint(0, len(synth_files)-1)
    bkg_idx = random.randint(0, len(bkg_files)-1)

    fish_img = Image.open(synth_files[fish_idx])
    png_info = fish_img.info
    bkg_img = Image.open(bkg_files[bkg_idx])

    rotation = random.sample([0, 90, 180, 270], 1)[0]
    x = random.randint(0, 720)
    y = random.randint(0, 360)

    scale = random.uniform(0.75, 1.25)

    img_size = list(fish_img.size)

    img_size[0] = img_size[0] * scale
    img_size[1] = img_size[1] * scale

    fish_img.thumbnail((int(img_size[0]), int(img_size[1])), Image.ANTIALIAS)

    bkg_img.paste(fish_img.rotate(rotation), (x,y))

    bkg_img.save('data/synthetics/generated/synth_fish' + str(i + 1000) + '.jpg')
