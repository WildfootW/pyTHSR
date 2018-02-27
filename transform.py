from functools import partial

import cv2
import numpy as np

_ones = partial(np.ones, dtype=np.uint8)

# @param im should be numpy array
def close_then_open(im, k=2):
    cl_kernel, op_kernel = _ones((k, k)), _ones((k, k))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cl_kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, op_kernel)
    return im

# pixel range: -1.0 ~ 1.0
def preprocess(imgs):
    shp = imgs.shape
    imgs = np.squeeze(imgs)
    
    # binarize
    threshold = 150
    imgs[imgs > threshold] = 255
    imgs[imgs <= threshold] = 0

    # denoise
    if len(imgs.shape) == 3:
        imgs = np.array(list(map(partial(close_then_open, k=3), imgs)))
        #for i in range(len(imgs)):
        #    imgs[i] = close_then_open(imgs[i], k=3)
    else:
        imgs = close_then_open(imgs, k=3)
    imgs = imgs.reshape(shp)
    return imgs / 127.5 - 1

def postprocess(imgs): 
    return ((imgs+1)*127.5).astype(np.uint8)
