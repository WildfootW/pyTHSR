
from functools import partial
from itertools import chain
try:
    from itertools import izip, izip_longest
except:
    izip = zip
    from itertools import zip_longest as izip_longest

import numpy as np

from core import CHAR_NUM, classes, class_num
from image import _read_img
from transform import preprocess, postprocess, onehot

def get_generator_batch(captcha_gen, label_gen, bsize, with_clean=False, mode='denoise'):
    captcha_create = partial(captcha_gen.create_THSR_captcha, with_clean=with_clean)
    
    labels = list( label_gen() for _ in range(bsize) )
    data_pair = list(map(captcha_create, labels))
    X, y = list(map(list, zip(*data_pair)))
    X = preprocess(np.array(list(map(lambda x: np.expand_dims(x, axis=2), X))))
    
    if mode == 'denoise':
        y = np.array(list(map(lambda x: np.expand_dims(x, axis=2), y)))
    elif mode == 'OCR':
        _y = onehot(labels)
        y = [_y[:, i] for i in range(4)]
    else:
        raise RuntimeError('Unsupport Batch Mode: "%s"' % mode)

    return labels, X, y


def parse_output(probs):
    assert len(probs) == CHAR_NUM
    pred_chars = list(map(lambda prob: classes[np.argmax(prob)], chain.from_iterable(zip(*probs))))
    return list(map(''.join, izip(*[chain(pred_chars)]*CHAR_NUM)))

