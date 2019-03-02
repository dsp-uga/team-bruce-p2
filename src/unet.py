"""
U-Net Model
This model is based on the implementation from
"https://github.com/zhixuhao/unet"
"""
from data_loader import DataLoader
from matplotlib import image as mpimg
import numpy as np
import os
from pathlib import Path
from urllib import request

from keras import backend as K
from keras import regularizers
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *


def frame_pad(image, req_shape):
    left_right_pad = int((req_shape[0] - image.shape[0])/2)
    top_bottom_pad = int((req_shape[1] - image.shape[1])/2)
    arr = np.pad(image, [(left_right_pad, left_right_pad), (top_bottom_pad, top_bottom_pad)], "reflect")
    return np.expand_dims(arr.reshape(arr.shape + (1,)), axis=0)


def get_images(filenames, key):
    images = []
    path = os.path.join(dl.dataset_folder, key)
    for file in filenames:
        if key == 'train' or key == 'test':
            images.append(mpimg.imread(os.path.join(path, 'data', file, "frame0000.png")))
        else:
            images.append(mpimg.imread(os.path.join(path, file + '.png')))
    return images


def image_loader(key):
    if key == 'train':
        path = os.path.join(dl.dataset_folder, key)
        filenames = dl.train_hashes
        images = get_images(filenames, key)
    elif key == 'test':
        path = os.path.join(dl.dataset_folder, key)
        filenames = dl.test_hashes
        images = get_images(filenames, key)
    elif key == 'masks':
        path = os.path.join(dl.dataset_folder, key)
        filenames = dl.train_hashes
        images = get_images(filenames, key)

    stacked_images = np.vstack([frame_pad(image, (640, 640)) for image in images])
    return stacked_images


def unet(pretrained_weights=None, input_size=(640, 640, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def main():
    dl = DataLoader()
    train_x = image_loader('train')
    test_x = image_loader('test')
    train_y = image_loader('masks')
    model = unet()
    model.fit([train_x], [train_y])
    prediction = model.predict([test_x])


if __name__ == '__main__':
    main()
