# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------


import keras.backend as K
import numpy as np
import tensorflow as tf


def expansion_block(x, t, filters, block_id):
    prefix = "block_{}_".format(block_id)
    total_filters = t * filters
    x = tf.keras.layers.Conv2D(
        total_filters, 1, padding="same", use_bias=False, name=prefix + "expand"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + "expand_bn")(x)
    x = tf.keras.layers.ReLU(6, name=prefix + "expand_relu")(x)

    return x


def depthwise_block(x, stride, block_id):
    prefix = "block_{}_".format(block_id)
    x = tf.keras.layers.DepthwiseConv2D(
        3,
        strides=(stride, stride),
        padding="same",
        use_bias=False,
        name=prefix + "depthwise_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + "dw_bn")(x)
    x = tf.keras.layers.ReLU(6, name=prefix + "dw_relu")(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = "block_{}_".format(block_id)
    x = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "compress",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + "compress_bn")(x)
    return x


def Bottleneck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = tf.keras.layers.add([x, y])
    return y


def MobileNet(input_shape=[64, 64, 3], scaling_coeff=1.0):
    t = 3
    input_img = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(
        input_img
    )
    x = tf.keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = tf.keras.layers.ReLU(6, name="conv1_relu")(x)
    # 17 Bottlenecks
    x = projection_block(x, out_channels=16, block_id=1)
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=12 * scaling_coeff,
        stride=2,
        block_id=2,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=12 * scaling_coeff,
        stride=1,
        block_id=3,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=16 * scaling_coeff,
        stride=2,
        block_id=4,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=16 * scaling_coeff,
        stride=1,
        block_id=5,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=16 * scaling_coeff,
        stride=1,
        block_id=6,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=32 * scaling_coeff,
        stride=2,
        block_id=7,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=32 * scaling_coeff,
        stride=1,
        block_id=8,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=32 * scaling_coeff,
        stride=1,
        block_id=9,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=32 * scaling_coeff,
        stride=1,
        block_id=10,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=48 * scaling_coeff,
        stride=1,
        block_id=11,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=48 * scaling_coeff,
        stride=1,
        block_id=12,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=48 * scaling_coeff,
        stride=1,
        block_id=13,
    )
    x = Bottleneck(
        x,
        t,
        x.shape[-1],
        80 * scaling_coeff,
        2,
        14,
    )
    x = Bottleneck(
        x,
        t,
        x.shape[-1],
        80 * scaling_coeff,
        1,
        15,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=80 * scaling_coeff,
        stride=1,
        block_id=16,
    )
    x = Bottleneck(
        x,
        t=t,
        filters=x.shape[-1],
        out_channels=160 * scaling_coeff,
        stride=1,
        block_id=17,
    )
    x = tf.keras.layers.Conv2D(
        filters=1280 * scaling_coeff,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="last_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="last_bn")(x)
    x = tf.keras.layers.ReLU(6, name="last_relu")(x)
    model = tf.keras.models.Model(input_img, x)
    return model
