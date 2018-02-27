#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Concatenate, GaussianNoise
from keras.models import Model, Sequential
from keras.preprocessing.image import NumpyArrayIterator

from core import HEIGHT, WIDTH, CHAR_NUM, class_num
from layer_utils import build_resnet_block

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
    nn = Dense(10, activation='selu')(nn)
    nn = GaussianNoise(0.1)(nn)
    char_outputs = [Dense(class_num, activation='softmax')(nn) for _ in range(CHAR_NUM)]
    #char_outputs = Concatenate(axis=1)()

    return Model(inputs=input_, outputs=char_outputs)
