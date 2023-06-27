# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------


import keras.backend as K
import numpy as np
import tensorflow as tf


def channel_shuffle(x, groups):
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups
    x = tf.keras.layers.Reshape([width, height, group_ch, groups])(x)
    x = tf.keras.layers.Permute([1, 2, 4, 3])(x)
    x = tf.keras.layers.Reshape([width, height, channels])(x)
    return x


def shuffle_unit(x, groups, channels, strides):
    y = x
    x = tf.keras.layers.Conv2D(
        channels // 4, kernel_size=1, strides=(1, 1), padding="same", groups=groups
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = channel_shuffle(x, groups)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=strides, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if strides == (2, 2):
        channels = channels - y.shape[-1]
    x = tf.keras.layers.Conv2D(
        channels, kernel_size=1, strides=(1, 1), padding="same", groups=groups
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides == (1, 1):
        x = tf.keras.layers.Add()([x, y])
    if strides == (2, 2):
        y = tf.keras.layers.AvgPool2D((3, 3), strides=(2, 2), padding="same")(y)
        x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.ReLU()(x)
    return x


def ShuffleNet(input_shape=[64, 64, 3], scaling_coefficient=1.0):
    start_channels = 128 * scaling_coefficient
    groups = 2
    input_img = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(
        24, kernel_size=3, strides=(2, 2), padding="same", use_bias=True
    )(input_img)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)
    repetitions = [3, 7]
    for i, repetition in enumerate(repetitions):
        channels = start_channels * (2**i)
        x = shuffle_unit(x, groups, channels, strides=(2, 2))
        for i in range(repetition):
            x = shuffle_unit(x, groups, channels, strides=(1, 1))
    model = tf.keras.models.Model(input_img, x)
    return model
