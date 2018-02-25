#!/usr/local/bin/python
from __future__ import print_function

import os
import sys
import argparse
from functools import partial, reduce
from itertools import izip, chain

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard, CSVLogger, LambdaCallback, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Concatenate
from keras.models import Model, Sequential
from keras.preprocessing.image import NumpyArrayIterator
from memory_profiler import profile

from layer_utils import build_resnet_block
from captcha.image import ImageCaptcha

WIDTH, HEIGHT, CHAR_NUM = 128, 48, 4
classes = u'ACFHKMNQPRTYZ234579'
class_num = len(classes)
EPOCHS, BATCHSIZE, STEP_PER_EP = 10, 32, 100
VIS_SIZE = 5
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

def simple_cnn():
    input_ = Input((HEIGHT, WIDTH, 1))
    channel_dim = 1
    
    # encode
    x = Conv2D(16, (3, 3), activation='selu', padding='same')(input_)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # transform (?
    for i in range(5):
        encoded = build_resnet_block(encoded, 8)

    # decode
    x = Conv2D(8, (3, 3), activation='selu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    outputs = Conv2D(channel_dim, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=input_, outputs=outputs)
    return model

def simple_ocr():

    input_ = Input((HEIGHT, WIDTH, 1))
    nn = Conv2D(2, (3, 3), activation='selu', padding='same')(input_)
    nn = BatchNormalization()(nn)
    nn = Conv2D(4, (3, 3), activation='selu', padding='same')(nn)
    nn = BatchNormalization()(nn)
    nn = Conv2D(8, (3, 3), activation='selu', padding='same')(nn)
    nn = BatchNormalization()(nn)
    nn = Conv2D(4, (3, 3), activation='selu', padding='same')(nn)
    nn = BatchNormalization()(nn)
    nn = Conv2D(2, (3, 3), activation='selu', padding='same')(nn)
    nn = Flatten()(nn)
    char_outputs = [Dense(class_num, activation='softmax')(nn) for _ in range(CHAR_NUM)]
    #char_outputs = Concatenate(axis=1)()

    return Model(inputs=input_, outputs=char_outputs)

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
        for i in range(len(imgs)):
            imgs[i] = close_then_open(imgs[i], k=3)
    else:
        imgs = close_then_open(imgs, k=3)
    imgs = imgs.reshape(shp)
    return imgs / 127.5 - 1
def postprocess(imgs):
    return ((imgs+1)*127.5).astype(np.uint8)

def valid_generator(imgs, labels):
    while True:
        rnd_idxs = np.random.choice(len(labels), BATCHSIZE)
        _data = imgs[rnd_idxs]
        _label = labels[rnd_idxs]
        yield (_data, [_label[:, i] for i in range(4)])

@profile
def data_generator(mode='denoise'):
    while True:
        labels = [ get_rnd_label() for _ in range(BATCHSIZE) ]
        data_pair = [ get_cap(labels[_]) for _ in range(BATCHSIZE) ]
        X, y = map(list, zip(*data_pair))
        X = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), X))))
        if mode == 'denoise':
            y = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), y))))
            yield (X, y)
        elif mode == 'OCR':
            y = np.zeros((BATCHSIZE, CHAR_NUM, class_num), dtype=np.int)
            for i, char in enumerate(chain.from_iterable(labels)):
                y[i//4, i%4, classes.find(char)] = 1
            yield (X, [y[:, 0], y[:, 1], y[:, 2], y[:, 3]])
        
def parse_arg():
    parser = argparse.ArgumentParser('Captcha Breaker')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train_ocr', action='store_true')
    parser.add_argument('--train_denoise', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64) 
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arg()

    # validation data
    labels = list(map(lambda p: os.path.splitext(os.path.basename(p))[0], os.listdir('dataset/real_cap')))
    label_str = ''.join(labels)
    raw_images = list(map(lambda p: _read_img(os.path.join('dataset/real_cap', p+'.bmp')), labels))
    raw_images = np.concatenate(raw_images, axis=0)
    valid_x = preprocess(raw_images)
    valid_y = np.zeros((len(labels), CHAR_NUM, class_num), dtype=np.int)
    for i, char in enumerate(chain.from_iterable(labels)):
        valid_y[i//4, i%4, classes.find(char)] = 1
    valid_gen = valid_generator(valid_x, valid_y)


    loss = 'mse'
    es = EarlyStopping(patience=5)
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
                workers=1, callbacks=[es, mdckpt], validation_data=valid_gen, validation_steps=5)
        ocr_model.save('ocr_model.h5')

    # visualize
    if True:
        before = valid_x[:VIS_SIZE]
        after = postprocess(model.predict(before))
    elif False:
        labels = [ get_rnd_label() for _ in range(VIS_SIZE) ]
        data_pair = [ get_cap(labels[_]) for _ in range(len(labels)) ]
        X, y = list(map(list, zip(*data_pair)))
        before = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), X))))
        ground = np.array(list(map(lambda x: np.expand_dims(x, axis=2), y)))
        after = postprocess(model.predict(before))

    pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*ocr_model.predict(before)))))
    # group every 4 chars
    pred_words = list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))
    print('\n'.join(map(str, zip(pred_words, labels))))

    compare_ones = np.vstack(np.squeeze(np.concatenate([before, after, ground], axis=2), axis=3))
    hist_images = Image.fromarray(compare_ones, mode='L')
    hist_images.show()


