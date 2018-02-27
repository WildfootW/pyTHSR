#!/usr/bin/env python

try:
    import cPickle as pickle
except:
    import _pickle as pickle
import os
import re
import json
from datetime import datetime
from itertools import chain
from functools import partial
from collections import OrderedDict

import PIL
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import Model

from core import WIDTH, HEIGHT, captcha, CHAR_NUM, classes, class_num
from image import from_directory
from handler import get_generator_batch
from transform import preprocess, postprocess, onehot
from simple_models import simple_cnn, simple_ocr

text_space = OrderedDict([
    ('rotate_range', [0, 10]), #  (-i, i) for i in range(1, 11)]),
    ('left_rate', [0.05, 0.2]),
    ('width_rate', [0.3, 0.5]),
    ('dy_rate_range', [0, 0.2]),  #(-i/20., i/20.) for i in range(4)]),
])

# all integers
curve_space = OrderedDict([
    ('rad_cent', [5., 6.]), 
    ('dx_range', [0, 5]), #(-i, i) for i in range(1, 6)]), 
    ('dy_cent', [2.5, 5.]), 
    ('width', [4, 6]), 
    ('rad_amp', [0.5, 1.5]),
    ('dy_amp', [1.5, 3]),
])

noise_space = OrderedDict([
    ('lamb', [0.1, 0.75]),
    ('std', [10, 64]),
    ('fn_prob', [0, 1]),
    #('fn', [partial(np.random.normal, loc=128)])
])


__kMap = { 'text_param': text_space.keys(), 'curve_param': curve_space.keys(), 'noise_param': noise_space.keys() }
EPOCHS, BATCHSIZE, STEP_PER_EP = 200, 32, 32
cap_gen = None
get_rnd_label = lambda: ''.join([classes[np.random.choice(class_num)] for _ in range(CHAR_NUM)])
labels, raw_imgs = from_directory('dataset/real_cap')
validation_data = preprocess(raw_imgs), onehot(labels)

def valid_generator(imgs, labels):
    while True:
        rnd_idxs = np.random.choice(len(labels), BATCHSIZE)
        _data, _label = imgs[rnd_idxs], labels[rnd_idxs]
        yield (_data, [_label[:, i] for i in range(CHAR_NUM)])

def data_generator(mode='denoise'):
    while True:
        labels, X, y = get_generator_batch(cap_gen, get_rnd_label, 
                BATCHSIZE, with_clean=True, mode=mode)
        yield (X, y)

def build_model():
    #denoise, ocr_layer = simple_cnn(), simple_ocr()
    #ocr_output = ocr_layer(denoise.outputs)
    ocr_model = simple_ocr() #Model(inputs=denoise.inputs, outputs=ocr_output)
    ocr_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return ocr_model

def modify_param(param):
    
    _param = dict( (__k, OrderedDict([(_k, param[_k]) for _k in _keys])) for __k, _keys in __kMap.items())
    
    for _k, _ks in __kMap.items():
        for k in _ks:
            if k.endswith('range'):
                _param[_k][k] = (-_param[_k][k], _param[_k][k])

    for _k, _ks in __kMap.items():
        for k in _ks:
            if k.endswith('cent'):
                _range, amp = k.replace('cent', 'range'), k.replace('cent', 'amp')
                _param[_k][_range] = ( _param[_k][k]-_param[_k][amp], _param[_k][k]+_param[_k][amp] )
                del _param[_k][k]
                del _param[_k][amp]

    for k, v in _param['curve_param'].items():
        if isinstance(v, tuple):
            _param['curve_param'][k] = tuple( int(elem) for elem in v )
        else:
            _param['curve_param'][k] = int(v)

    if _param['noise_param']['fn_prob'] > 0.5:
        _param['noise_param']['fn'] = partial(np.random.normal, loc=128)
    else:
        _param['noise_param']['fn'] = partial(np.random.poisson, lam=1.0)
    del _param['noise_param']['fn_prob']

    return _param

def evaluate(no_record=False, **param):

    global cap_gen, memory_df
    _param = modify_param(param)
    cap_gen = captcha(**_param)
    train_gen = data_generator('OCR')
    valid_gen = valid_generator(*validation_data)
    p_str = json.dumps(param)  # for bo_memory usage

    with tf.Session() as sess:
        K.set_session(sess)
        ocr_model = build_model()
        hist = ocr_model.fit_generator(train_gen, epochs=EPOCHS, steps_per_epoch=STEP_PER_EP,
                validation_data=valid_gen, validation_steps=STEP_PER_EP, verbose=0)

        # get score by logs ( train/valid correlation )
        loss, val_loss = hist.history['loss'], hist.history['val_loss']
        corr = np.corrcoef(loss, val_loss)[0][1]

        plt.clf()
        plt.plot(range(EPOCHS), loss, label='loss')
        plt.plot(range(EPOCHS), val_loss, label='val_loss')
        plt.legend()

    K.clear_session()

    if no_record:
        if abs(corr-best_score) > 0.05:
            print('Reconstruct score not match, curr: %f, best: %f' \
                    % (corr, best_score))
            return evaluate(no_record, **param)

        plt.savefig('best.png')
        PIL.Image.open('best.png').show()
        print(corr)
        ocr_model.save('BestSimpleModel.h5')
    else:
        fname = re.sub(r'[-:. ]', 'I', str(datetime.now()))
        plt.savefig(fname+'.png')
        memory_df = memory_df.append(dict(Target=corr, **param), ignore_index=True)

        if len(memory_df) >= 10:     # flush
            memory_df.to_csv('bo_memory', mode='a', index=False)
            memory_df = pd.DataFrame(columns=memory_df.columns)

        with open('file_mapping.txt', 'a') as f:
            f.write('%s, %s\n' % (fname, p_str))

    return corr

if __name__ == '__main__':

    if not os.path.exists('bayes_model'):
        # use bayesian method to find best parameter
        space = OrderedDict(chain(text_space.items(), curve_space.items(), noise_space.items()))
        bo = BayesianOptimization(evaluate, space)
        memory_df = pd.DataFrame([], columns=['target'] + bo.space.keys)
        if os.path.exists('bo_memory'):
            bo.initialize_df(pd.read_csv('bo_memory'))
        init_num = max(0, 20 - len(bo.x_init))
        bo.maximize(init_points=init_num, n_iter=90, kappa=2)

        with open('bayes_model', 'wb') as f:
            pickle.dump(bo, f)

        bo.points_to_csv('steps.csv')

    else:
        with open('bayes_model', 'rb') as f:
            bo = pickle.load(f)

        best_score = bo.res['max']['max_val']
        evaluate(no_record=True, **bo.res['max']['max_params'])

