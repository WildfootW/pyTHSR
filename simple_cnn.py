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
from keras.models import Model
from keras.callbacks import TensorBoard, CSVLogger, LambdaCallback, Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler

from transform import preprocess, postprocess, onehot
from core import WIDTH, HEIGHT, CHAR_NUM, classes, class_num, captcha
from image import from_directory
from handler import get_generator_batch
from simple_models import simple_cnn, simple_ocr

EPOCHS, BATCHSIZE, STEP_PER_EP = 10, 32, 20
VIS_SIZE = 5
cap_gen = captcha(curve_width=6)
get_cap = partial(cap_gen.create_THSR_captcha, with_clean=True)
get_rnd_label = lambda: ''.join([classes[np.random.choice(class_num)] for _ in range(CHAR_NUM)])

def lr_sched(ep): 
    return 1e-2 - 1e-4*0.01*ep

def valid_generator(imgs, labels):
    while True:
        rnd_idxs = np.random.choice(len(labels), len(labels))
        it = 0
        while it + BATCHSIZE < len(labels):
            rnd_id = rnd_idxs[it:it+BATCHSIZE]
            _data, _label = imgs[rnd_id], labels[rnd_id]
            yield (_data, [_label[:, i] for i in range(CHAR_NUM)])
            it += BATCHSIZE

def data_generator(mode='denoise'):
    while True:
        labels, X, y = get_generator_batch(cap_gen, get_rnd_label, 
                BATCHSIZE, with_clean=True, mode=mode)
        yield (X, y)

def parse_arg():
    parser = argparse.ArgumentParser('Captcha Breaker')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train_ocr', action='store_true')
    parser.add_argument('--train_denoise', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64) 
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arg()
    BATCHSIZE = args.batch_size

    # validation data
    if not os.path.exists(os.path.join('dataset', 'real_cap')):
        os.system('cd dataset; tar -xvf real_cap.tar.gz > /dev/null')

    labels, raw_images = from_directory(os.path.join('dataset', 'real_cap'))
    valid_x = preprocess(raw_images)
    valid_y = onehot(labels)
    valid_gen = valid_generator(valid_x, valid_y)


    loss = 'mse'
    es = EarlyStopping(patience=5)
    lrs = LearningRateScheduler(lr_sched)
    mdckpt = ModelCheckpoint('weights.{loss}.hdf5'.format(loss=loss), save_best_only=True, save_weights_only=True)
    model, ocr_layer = simple_cnn(), simple_ocr()

    freeze_layers = (model.layers if args.train_ocr else [])
    for layer in freeze_layers:
        layer.trainable = False

    ocr_output = ocr_layer(model.outputs)
    ocr_model = Model(inputs=model.inputs, outputs=ocr_output)

    if args.load:
        if os.path.exists('ocr_model.h5'):
            ocr_model.load_weights('ocr_model.h5')
        if os.path.exists('denoise.h5'):
            model.load_weights('denoise.h5')

    freeze_layers = (ocr_layer.layers if args.train_denoise else [])
    for layer in freeze_layers:
        layer.trainable = False

    if args.train_denoise:
        model.compile(optimizer='adam', loss=loss)
        model.fit_generator(data_generator(), epochs=EPOCHS, steps_per_epoch=STEP_PER_EP,
                workers=1, callbacks=[es, mdckpt])
        model.save('denoise.h5')

    if args.train_ocr:
        ocr_model.compile(optimizer='adam', loss='categorical_crossentropy')
        ocr_model.fit_generator(data_generator('OCR'), epochs=EPOCHS, steps_per_epoch=STEP_PER_EP,
                workers=1, callbacks=[es, mdckpt], validation_data=valid_gen, validation_steps=STEP_PER_EP)
        ocr_model.save('ocr_model.h5')

    # visualize
    before = valid_x[np.random.choice(len(valid_x), VIS_SIZE)]
    ground = postprocess(before)
    after = postprocess(model.predict(before))
    combined_valid = np.concatenate([postprocess(before), after, ground], axis=2)

    _labels, before, ground = get_generator_batch(cap_gen, get_rnd_label, VIS_SIZE, with_clean=True)
    after = postprocess(model.predict(before))
    combined_gened = np.concatenate([postprocess(before), after, ground], axis=2)

    #pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*ocr_model.predict(before)))))
    # group every 4 chars
    #pred_words = list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))
    #print('\n'.join(map(str, zip(pred_words, labels))))

    compare_ones = np.vstack(np.squeeze(np.r_[combined_valid, combined_gened], axis=3))
    hist_images = Image.fromarray(compare_ones, mode='L')
    hist_images.show()


