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
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard, CSVLogger, LambdaCallback, Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Concatenate
from keras.models import Model, Sequential
from keras.preprocessing.image import NumpyArrayIterator
from memory_profiler import profile

from layer_utils import build_resnet_block
from transform import preprocess, postprocess
from core import WIDTH, HEIGHT, CHAR_NUM, classes, class_num, captcha
from image import _read_img

EPOCHS, BATCHSIZE, STEP_PER_EP = 10, 32, 20
VIS_SIZE = 5
cap_gen = captcha()
get_cap = partial(cap_gen.create_THSR_captcha, color='black', background='#fff', pen_size=6, with_clean=True)
get_rnd_label = lambda: ''.join([classes[np.random.choice(class_num)] for _ in range(4)])

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

def valid_generator(imgs, labels):
    while True:
        rnd_idxs = np.random.choice(len(labels), BATCHSIZE)
        _data, _label = imgs[rnd_idxs], labels[rnd_idxs]
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
            yield (X, [y[:, i] for i in range(4)])
        
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
    if not os.path.exists('dataset/real_cap'):
        os.system('cd dataset; tar -xvf real_cap.tar.gz > /dev/null')

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
                workers=1, callbacks=[es, mdckpt], validation_data=valid_gen, validation_steps=STEP_PER_EP)
        ocr_model.save('ocr_model.h5')

    # visualize
    before = valid_x[np.random.choice(len(valid_x), VIS_SIZE)]
    ground = postprocess(before)
    after = postprocess(model.predict(before))
    combined_valid = np.concatenate([postprocess(before), after, ground], axis=2)

    labels = [ get_rnd_label() for _ in range(VIS_SIZE) ]
    data_pair = [ get_cap(labels[_]) for _ in range(len(labels)) ]
    X, y = list(map(list, zip(*data_pair)))
    before = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), X))))
    ground = np.array(list(map(lambda x: np.expand_dims(x, axis=2), y)))
    after = postprocess(model.predict(before))
    combined_gened = np.concatenate([postprocess(before), after, ground], axis=2)

    #pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*ocr_model.predict(before)))))
    # group every 4 chars
    #pred_words = list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))
    #print('\n'.join(map(str, zip(pred_words, labels))))

    compare_ones = np.vstack(np.squeeze(np.r_[combined_valid, combined_gened], axis=3))
    hist_images = Image.fromarray(compare_ones, mode='L')
    hist_images.show()


