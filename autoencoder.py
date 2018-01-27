
import os
import sys
import atexit
import argparse
from glob import glob
from time import sleep
from functools import partial
try:    # py3
    from itertools import filterfalse
except: # py2
    from itertools import ifilterfalse as filterfalse

import numpy as np
from keras.callbacks import TensorBoard, CSVLogger, LambdaCallback, Callback
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import Iterator, load_img, img_to_array
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from layer_utils import build_resnet_block, randReadImg

# Constant
WIDTH, HEIGHT = 128, 48

def network(ishape):
    
    input_img = Input(shape=ishape)
    channel_dim = ishape[-1]

    # encode
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # transform (?
    for i in range(5):
        encoded = build_resnet_block(encoded, 8)

    # decode
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channel_dim, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder

class ImageLogger(Callback):

    def __init__(self, sample_image):
        super(ImageLogger, self).__init__()
        self.sample_images = sample_image
        print('Prepare %d images to visualize in ImageLogger' % len(self.sample_images))

    def on_train_begin(self, logs={}):
        self.losses = []
        self.outputs = []

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.outputs.append(self.model.predict(self.sample_images))

# My implementation of "https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py"
def _get_valid_files_in_dir(directory, white_list_formats, follow_links):
    """get  files with extension in `white_list_formats` contained in directory.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        follow_links: boolean.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    samples = []
    for typ in white_list_formats:
        samples.extend(glob(os.path.join(directory, '*.'+typ)))
    if not follow_links:
        samples = list(filterfalse(os.path.islink, samples))
    return samples

class SimpleIterator(Iterator):

    def __init__(self, directory, datagen,
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=32,
            shuffle=True,
            seed=None,
            data_format=None,
            follow_link=False,
            interpolation='nearest'):
        self.data_format = data_format or K.image_data_format()
        
        self.directory = directory
        self.data_generator = datagen
        self.target_size = target_size
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = self.target_size + (1,)
        
        if self.data_format == 'channel_first':
            self.image_shape = (self.image_shape[-1],) + self.target_size
        
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}        
        self.filenames = _get_valid_files_in_dir(self.directory, 
                white_list_formats, follow_link)
        self.samples = len(self.filenames)
        self.interpolation = interpolation
    
        super(SimpleIterator, self).__init__(
                self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x *= self.data_generator.rescale
            batch_x[i] = x

        # label is original image
        batch_y = batch_x.copy()

        return batch_x, batch_y


    def next(self):
        """For python 2.x.
        
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array) 


class DataGenerator(object):

    def __init__(self, rescale=None):

        self.rescale = rescale or 1.

    def flow_from_dir(self, path, 
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=32,
            shuffle=True,
            seed=None,
            data_format=None,
            follow_link=False):
        return SimpleIterator(
                path, 
                self,
                target_size=target_size, 
                color_mode=color_mode, 
                batch_size=batch_size, 
                shuffle=shuffle, seed=seed,
                data_format=data_format,
                follow_link=follow_link)

    def post_process(self, data):
        return data / self.rescale

    def to_gray(self, data, func=np.vstack):
        data = self.post_process(data)
        data = data.astype(np.uint8)
        #data = np.swapaxes(data, 1, 2)  # for pillow image format(?
        data = np.squeeze(func(data))
        return data

@atexit.register
def cleanup():
    K.clear_session()
    sleep(.5)
    print('clean')

def parse_args():
    parser = argparse.ArgumentParser('AutoEncoder')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64) 
    return parser.parse_args()

if __name__ == '__main__':

    directory = 'dataset/denoise/noise_caps'
    color_mode = 'grayscale'
    pillow_mode = 'L' if color_mode == 'grayscale' else 'rgb'
    args = parse_args()

    datagen = DataGenerator(rescale=1./255)
    flows = datagen.flow_from_dir(
        directory, 
        target_size=(HEIGHT, WIDTH), 
        color_mode=color_mode,
        batch_size=args.batch_size,
        shuffle=True)

    sample_image = randReadImg(directory, 5, shp=flows.image_shape, absolute_path=True)
    imgLogger = ImageLogger(sample_image=sample_image)

    channel_dim = 3 if flows.color_mode == 'rgb' else 1
    model = network((HEIGHT, WIDTH, channel_dim))

    if args.load:
        model.load_weights('encoder.h5')
    
    step_per_ep = int(flows.samples/flows.batch_size)
    
    if args.train:
        history = model.fit_generator(
            flows, steps_per_epoch=step_per_ep, epochs=10, callbacks=[
                TensorBoard(log_dir='/tmp/autoencoder'), imgLogger])
    else:
        history = None

    model.save('encoder.h5')

    recon_image = model.predict_generator(flows, steps=1)[:10]
    recon_image = datagen.to_gray(recon_image)
    im = Image.fromarray(recon_image, mode=pillow_mode)
    im.show()

    if args.train:
        images = []
        for ep, output in enumerate(imgLogger.outputs):
            gray_im = datagen.to_gray(output, func=np.hstack)
            images.append(gray_im)
        hist_images = np.vstack(images)
        hist_images = Image.fromarray(hist_images, mode=pillow_mode)
        hist_images.show()

