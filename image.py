import os
from glob import glob
try:
    from StringIO import StringIO  # py2
except:
    from io import BytesIO as StringIO  # py3

import numpy as np
from PIL import Image

from core import WIDTH, HEIGHT

def _read_img(path):
    im = Image.open(path)
    im.load()
    imnpy = np.asarray(_preprocess(im))
    return imnpy.reshape((1, HEIGHT, WIDTH, 1))

def Byte2Img(byte):
    img = Image.open(StringIO(byte))
    return _preprocess(img)

def _preprocess(im):
    '''PIL.Image -> GrayScale -> (WIDTH, HEIGHT) '''
    return im.convert('L').resize((WIDTH, HEIGHT), Image.BILINEAR)

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

