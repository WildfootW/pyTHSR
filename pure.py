#!/usr/local/bin/python
from __future__ import print_function

import os
import sys
import argparse
from functools import partial, reduce
from itertools import chain
try:
    from itertools import izip
except:
    izip = zip

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, LambdaCallback, Callback, EarlyStopping, ModelCheckpoint

from transform import preprocess, postprocess, onehot
from core import WIDTH, HEIGHT, CHAR_NUM, classes, class_num, captcha
from image import from_directory
from handler import get_generator_batch
from simple_models import simple_ocr, simple_cnn

EPOCHS, BATCHSIZE = 300, 32
VIS_SIZE = 5

def parse_arg():
    parser = argparse.ArgumentParser('Captcha Breaker')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64) 
    return parser.parse_args()

def predict_chars(x_prob, y_words):
    pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*x_prob))))
    # group every 4 chars
    pred_words = list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))
    return '\n'.join(map(str, zip(pred_words, y_words)))

if __name__ == '__main__':

    np.random.seed(666)
    args = parse_arg()
    BATCHSIZE = args.batch_size

    # validation data
    if not os.path.exists(os.path.join('dataset', 'real_cap')):
        os.system('cd dataset; tar -xvf real_cap.tar.gz > /dev/null')

    labels, raw_images = from_directory(os.path.join('dataset', 'real_cap'))
    X = preprocess(raw_images)
    y = onehot(labels)
    trX, vaX, trY, vaY, trL, vaL = train_test_split(X, y, np.asarray(labels), test_size=0.1, shuffle=True)
    trY = [ trY[:, i] for i in range(CHAR_NUM) ]
    vaY = [ vaY[:, i] for i in range(CHAR_NUM) ]

    ckpt_name = os.path.splitext(__file__)[0] + '.best.h5'
    model_name = os.path.splitext(__file__)[0] + '.h5'
    es = EarlyStopping(patience=10)
    mdckpt = ModelCheckpoint(ckpt_name, save_best_only=True, save_weights_only=True)
    deno, ocr = simple_cnn(), simple_ocr()
    #ocr_out = ocr(deno.outputs)
    #ocr_model = Model(inputs=deno.inputs, outputs=ocr_out)
    ocr_model = ocr

    if args.load:
        if os.path.exists(model_name):
            ocr_model.load_weights(model_name)

    if args.train:
        ocr_model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy')
        ocr_model.fit(trX, trY, epochs=EPOCHS, callbacks=[es, mdckpt], 
                validation_data=(vaX, vaY))
        ocr_model.load_weights(ckpt_name)
        ocr_model.save(model_name)

    rnd_idxes = np.random.choice(len(vaX), VIS_SIZE)
    trImg, trLab = trX[rnd_idxes], trL[rnd_idxes]
    vaImg, vaLab = vaX[rnd_idxes], vaL[rnd_idxes]

    tr_res = predict_chars(ocr_model.predict(trImg), trLab)
    va_res = predict_chars(ocr_model.predict(vaImg), vaLab)
    print(''' TrainSet 90%  -- \n{tr} \nValidSet 10% -- \n{va} \n'''.format(tr=tr_res, va=va_res))

    compare_ones = np.vstack(np.squeeze(np.r_[postprocess(trImg), postprocess(vaImg)], axis=3))
    hist_images = Image.fromarray(compare_ones, mode='L')
    hist_images.show()


