"""
U-Net Model

This model is based on the implementation from
"https://github.com/zhixuhao/unet"

The Batch Normalisation is based on the
"Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift" by Sergey Ioffe, Christian Szegedy
"""
from src.data_loader import DataLoader
from matplotlib import image as mpimg
import numpy as np
import os
from keras import backend as K
from keras import regularizers
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

dl = DataLoader()


def dice_loss_function(y_true, y_pred, smooth=1):
    """
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
    :y_true: ground truth
    :y_pred: prediction
    :return: the negative of dice_loss_function
    """
    return 0 - dice_loss_function(y_true, y_pred)


def frame_pad(image, req_shape):
    """
    This function pads the image to required shape
    :image: image that is to be padded
    :req_shape: shape of the final_image
    :return: frame of shape (1, final_image.shape, 1)
    """
    left_right_pad = int((req_shape[0] - image.shape[0])/2)
    top_bottom_pad = int((req_shape[1] - image.shape[1])/2)
    arr = np.pad(image, [(left_right_pad, left_right_pad), (top_bottom_pad, top_bottom_pad)], "reflect")
    return np.expand_dims(arr.reshape(arr.shape + (1,)), axis=0)


def get_images(filenames, key):
    """
    This function reads all the images based on the key as a list of arrays
    :key: The key used to specify either train, mask or test images to be read
    :filenames: The filenames of files from which images are to be read
    :return: list of arrays
    """
    images = []
    path = os.path.join(dl.dataset_folder, key)
    for file in filenames:
        if key == 'train' or key == 'test':
            images.append(mpimg.imread(os.path.join(path, 'data', file, "frame0000.png")))
        else:
            images.append(mpimg.imread(os.path.join(path, file + '.png')))
    return images


def image_loader(key):
    """
    This function loads all the images,pads, reshapes and
    expands the dimensions of image
    :key: The key used to specify either train, mask or test images to be read
    :return: a stack of image arrays of same shape
    """
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


def unet_model(l2_reg=0.0002, lr=1e-5, kernel_size=3, dropout_rate=0.3, input_shape=(640, 640, 1)):
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
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv1)
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
    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg))(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=dice_loss_function, metrics=[dice_coefficient])
    return model


def UNet():
    train_x = image_loader('train')
    test_x = image_loader('test')
    train_y = image_loader('masks')
    model = unet_model()
    model.fit([train_x], [train_y])
    prediction = model.predict([test_x])
    for i in range(len(prediction)):
        filename = "results/experiment1/prediction" + str(i)
        np.save(filename, prediction[i])
