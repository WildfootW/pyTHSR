#!/usr/local/bin/python
from __future__ import print_function

import os
import sys
from functools import partial
from itertools import chain
try:
    from itertools import izip, izip_longest
except:
    izip = zip
    from itertools import zip_longest

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model

from core import captcha, WIDTH, HEIGHT, class_num, classes, CHAR_NUM
from image import _read_img
from transform import preprocess, postprocess
from captcha.image import ImageCaptcha
from layer_utils import InstanceNormalization2D

_ones = partial(np.ones, dtype=np.uint8)
cap_gen = captcha(curve_width=5)
get_cap = partial(cap_gen.create_THSR_captcha, with_clean=True)
get_rnd_label = lambda: ''.join([classes[np.random.choice(class_num)] for _ in range(4)])

def parse_output(probs):
    assert len(probs) == CHAR_NUM
    pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*probs))))
    return list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))

if __name__ == '__main__':

    if not os.path.exists('dataset/real_cap'):
        os.system('cd dataset; tar -xvf real_cap.tar.gz')

    model = load_model('BestSimpleModel.h5', custom_objects=dict(InstanceNormalization2D=InstanceNormalization2D))

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

