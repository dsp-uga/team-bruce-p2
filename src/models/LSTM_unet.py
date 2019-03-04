"""
    Author: Yang Shi
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import regularizers
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

MASK_DIR = "../masks/"
DATA_DIR = "../data/"
TRAIN_FILE = "../train.txt"
TEST_FILE = "../test.txt"
RESULT_DIR = "../result/"

def read_file(file_name):
    f = open(file_name, "r")
    return f.read().split()

def get_image(hash_code):
    '''
        input: hash_code for the train sample
        output: images: 3-D np array of the train sample, the first dimension is the frame.
        '''
    image_folder = DATA_DIR + hash_code
    image_frames = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = []
    for frame in image_frames:
        images.append(cv.cvtColor(cv.imread(os.path.join(image_folder, frame)), cv.COLOR_BGR2GRAY))
    images = np.array(images)
    return images

def dice_loss_function(y_true, y_pred, smooth=1):
    """
        This is identical to unet.py
        This function is implementation of Sørensen–Dice coefficient
        :y_true: ground truth
        :y_pred: prediction
        :smooth: smoothing parameter to negate division by zero error
        :return: dice loss
        """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient(y_true, y_pred):
    """
        This is identical to unet.py
        This function is implementation of Sørensen–Dice coefficient
        :y_true: ground truth
        :y_pred: prediction
        :return: the negative of dice_loss_function
        """
    return 0 - dice_loss_function(y_true, y_pred)

def lstm_unet(l2_reg=0.0002, lr=1e-5, kernel_size=3, dropout_rate=0.3, input_shape=(100, 256, 256,1)):
    """
        This model is based on the implementation from
        "https://github.com/zhixuhao/unet"
        
        U-net model with Batch Normalisation
        
        :l2_reg: penalty on layer parameters applied during optimization
        :lr: learning rate for the optimizer
        :kernal_size: length of convolutional window size
        :dropout_rate: Random neuron dropping rate to prevent overfitting
        :input_shape: a 3d tensor with shape (image_width,image_height,channels)
        
        :Return: a Unet Model
        """
    
    inputs = Input(input_shape)
    LSTM1 = ConvLSTM2D(32, (kernel_size, kernel_size), padding='same', activation='tanh')(inputs)
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(LSTM1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)
    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)
    conv3 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)
    conv4 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv5)
    conv5 = BatchNormalization()(conv5)
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], name='up6', axis=3)
    up6 = Dropout(dropout_rate)(up6)
    conv6 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], name='up7', axis=3)
    up7 = Dropout(dropout_rate)(up7)
    conv7 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], name='up8', axis=3)
    up8 = Dropout(dropout_rate)(up8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], name='up9', axis=3)
    up9 = Dropout(dropout_rate)(up9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=dice_loss_function, metrics=[dice_coefficient])
    return model

def main():
    train_hash = read_file(TRAIN_FILE)
    small_train_hash = train_hash[:5]

    # Read in as pandas dataframe
    train_df = pd.DataFrame(train_hash,columns=['Hash_code'])
    train_df.mask = [cv.imread(MASK_DIR + x + ".png") for x in train_df.Hash_code]
    train_df.images = [get_image(x) for x in train_df.Hash_code]

    # Resizing and recording dimensions
    train_df.img_dimensions = [x.shape for x in train_df.images]
    train_df.resized_imgs = [np.array([cv.resize(f,(256,256)) for f in x]) for x in train_df.images]
    train_df.resized_masks = [np.array(cv.resize(x,(256,256))) for x in train_df.mask]

    X_train, X_test, y_train, y_test = train_test_split(train_df.resized_imgs, train_df.resized_masks, test_size=0.33)

    # Building model
    model = lstm_unet()
    model.fit(np.stack(X_train,axis=0)[...,np.newaxis], np.stack(np.mean(y_train,axis=3),axis=0)[...,np.newaxis])

    # Making predictions
    prediction = model.predict(np.stack(X_train,axis=0)[...,np.newaxis])

if __name__ == '__main__':
    main()
