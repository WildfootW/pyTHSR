#!/usr/local/bin/python
from __future__ import print_function

import os
import sys
from functools import partial
from itertools import chain, izip, izip_longest

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model

from captcha.image import ImageCaptcha
from layer_utils import InstanceNormalization2D

WIDTH, HEIGHT, CHAR_NUM = 128, 48, 4
classes = u'ACFHKMNQPRTYZ234579'
class_num = len(classes)
font = 'fonts/MyriadPro-Semibold.otf'
_ones = partial(np.ones, dtype=np.uint8)
cap_gen = ImageCaptcha(width=WIDTH, height=HEIGHT, fonts=[font], font_sizes=[42,])
get_cap = partial(cap_gen.create_THSR_captcha, color='black', background='#fff', pen_size=5, with_clean=True)
get_rnd_label = lambda: ''.join([classes[np.random.choice(class_num)] for _ in range(4)])

def _read_img(path):
    im = Image.open(path).convert('L')
    im.load()
    im = im.resize((WIDTH, HEIGHT), Image.BILINEAR)
    return np.asarray(im).reshape((1, HEIGHT, WIDTH, 1))

# @param im should be numpy array
def close_then_open(im, k=2):
    cl_kernel, op_kernel = _ones((k, k)), _ones((k, k))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cl_kernel)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, op_kernel)
    return im

def preprocess(imgs):
    shp = imgs.shape
    imgs = np.squeeze(imgs)

    # binarize
    threshold = 150
    imgs[imgs > threshold] = 255
    imgs[imgs <= threshold] = 0

    if len(imgs.shape) == 3:
        for i in range(len(imgs)):
            imgs[i] = close_then_open(imgs[i], k=3)
    else:
        imgs = close_then_open(imgs, k=3)
    imgs = imgs.reshape(shp)
    return imgs / 127.5 - 1
def postprocess(imgs):
    return ((imgs+1)*127.5).astype(np.uint8)
def parse_output(probs):
    assert len(probs) == CHAR_NUM
    pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*probs))))
    return list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))

if __name__ == '__main__':

    if not os.path.exists('dataset/real_cap'):
        os.system('cd dataset; tar -xvf real_cap.tar.gz')

    model = load_model('ocr_model.h5', custom_objects=dict(InstanceNormalization2D=InstanceNormalization2D))

    labels = list(map(lambda p: os.path.splitext(os.path.basename(p))[0], os.listdir('dataset/real_cap')[:10]))
    label_str = ''.join(labels)
    raw_images = list(map(lambda p: _read_img(os.path.join('dataset/real_cap', p+'.bmp')), labels))
    raw_images = np.concatenate(raw_images, axis=0)
    imgs = preprocess(raw_images)

    data_pair = [ get_cap(labels[_]) for _ in range(len(labels)) ]
    X, _ = map(list, zip(*data_pair))
    gen_imgs = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), X))))

    if __debug__:
        gen_imgs = postprocess(imgs)
        im = Image.fromarray(gen_imgs.reshape(-1, WIDTH), mode='L')
        im.show()
        exit()

    logs = pd.DataFrame(data=labels, columns=['Ground'])

    # Test with hand-labeled THSR Captcha
    logs['THSR Cap'] = parse_output(model.predict(imgs))

    # Test with cap generator
    logs['generated'] = parse_output(model.predict(gen_imgs))

    logs.loc['word score'] = [ float(np.sum(logs[col][:len(labels)] == labels))/len(labels) for col in logs.columns ]
    logs.loc['char score'] = [ float(sum([int(c0==c1) for c0, c1 in izip_longest(''.join(logs[col][:len(labels)]), label_str)]))/(CHAR_NUM*len(labels)) for col in logs.columns ]
    logs.to_csv('exp_result.csv')

