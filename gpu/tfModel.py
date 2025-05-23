# coding: utf-8
'''
    - "ZF_UNET_224" Model based on UNET code from following paper: https://arxiv.org/abs/1505.04597
    - This model used to get 2nd place in DSTL competition: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
    - For training used DICE coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    - Input shape for model is 224x224 (the same as for other popular CNNs like VGG or ResNet)
    - It has 3 input channels (to process standard RGB (BGR) images). You can change it with variable "INPUT_CHANNELS"
    - It trained on random image generator with random light shapes (ellipses) on dark background with noise (< 10%).
    - In most cases model ZF_UNET_224 is ok to be used without pretrained weights.
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization
from keras.layers import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers import concatenate
from tensorflow.keras.utils import get_file
import os#
from tensorflow.keras.utils import plot_model
import numpy as np
# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1
# Pretrained weights
ZF_UNET_224_WEIGHT_PATH = 'https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/releases/download/v1.0/zf_unet_224.h5'

def savenpy(tensor,name):
  flattened = np.array(tensor).flatten()
  npy_file_path = f"{name}.npy"
  np.save(os.path.join("results",npy_file_path), flattened)

def preprocess_input(x):
    x = x.astype(np.float64)
    x /= 256
    x -= 0.5
    return x

def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    if K.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def ZF_UNET_224(dropout_val=0.2, weights=None):
    if K.image_data_format() == 'channels_first':
        inputs = Input((INPUT_CHANNELS, 224, 224))
        axis = 1
    else:
        inputs = Input((224, 224, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)


    conv_112 = double_conv_layer(pool_112, 2*filters)

    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)


    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)

    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)

    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)

    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)

    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)

    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)

    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)

    up_conv_224 = double_conv_layer(up_224, filters, dropout_val)


    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)

    conv_final = Activation('sigmoid')(conv_final)
    model = Model(inputs, conv_final, name="ZF_UNET_224")
    if weights == 'generator' and axis == 3 and INPUT_CHANNELS == 3 and OUTPUT_MASK_CHANNELS == 1:
        weights_path = get_file(
            'zf_unet_224_weights_tf_dim_ordering_tf_generator.h5',
            ZF_UNET_224_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='203146f209baf34ac0d793e1691f1ab7')
        model.load_weights(weights_path)

    return model
