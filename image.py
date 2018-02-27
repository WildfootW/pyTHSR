import os
from glob import glob

import numpy as np
from PIL import Image

from core import WIDTH, HEIGHT

def _read_img(path):
    im = Image.open(path).convert('L')
    im.load()
    im = im.resize((WIDTH, HEIGHT), Image.BILINEAR)
    return np.asarray(im).reshape((1, HEIGHT, WIDTH, 1))

def from_directory(dirname):
    types = ('bmp',)
    files = []
    for typ in types:
        files += glob(os.path.join(dirname, '*.'+typ))
    labels = list(map(lambda p: os.path.splitext(os.path.basename(p))[0], files))
    raw_images = np.zeros((len(files), HEIGHT, WIDTH, 1))
    for i, fname in enumerate(files):
        raw_images[i] = _read_img(fname)[0]
    return labels, raw_images

