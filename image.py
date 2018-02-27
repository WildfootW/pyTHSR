import numpy as np
from PIL import Image
from core import WIDTH, HEIGHT

def _read_img(path):
    im = Image.open(path).convert('L')
    im.load()
    im = im.resize((WIDTH, HEIGHT), Image.BILINEAR)
    return np.asarray(im).reshape((1, HEIGHT, WIDTH, 1))
