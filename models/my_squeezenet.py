# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------
"""
SqueezeNet implemented in Keras
This implementation is based on the original paper.
# References
- [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [GitHub](https://github.com/DeepScale/SqueezeNet)
@author: Christopher Masch
"""

import keras.backend as K
import numpy as np
import tensorflow as tf

__version__ = "0.0.1"


def SqueezeNet(
    input_shape: list = [64, 64, 3], dropout_rate=None, compression: float = 1.0
):
    """
    Creating a SqueezeNet of version 1.1

    Arguments:
        input_shape  : shape of the input images e.g. (224,224,3)
        dropout_rate : defines the dropout rate that is accomplished after last fire module (default: None)
        compression  : reduce the number of feature-maps

    Returns:
        Model        : Keras model instance
    """

    input_img = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        int(64 * compression),
        (3, 3),
        activation="relu",
        strides=(2, 2),
        padding="same",
        name="conv1",
    )(input_img)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name="maxpool1")(x)

    x = create_fire_module(x, int(8 * compression), name="fire2")
    x = create_fire_module(x, int(16 * compression), name="fire3")

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name="maxpool3")(x)

    x = create_fire_module(x, int(32 * compression), name="fire4")
    x = create_fire_module(x, int(32 * compression), name="fire5")

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name="maxpool5")(x)

    x = create_fire_module(x, int(48 * compression), name="fire6")
    x = create_fire_module(x, int(48 * compression), name="fire7")
    x = create_fire_module(x, int(64 * compression), name="fire8")
    x = create_fire_module(x, int(64 * compression), name="fire9")

    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    return tf.keras.models.Model(inputs=input_img, outputs=x)


def create_fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    """
    Creates a fire module

    Arguments:
        x                 : input
        nb_squeeze_filter : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
        use_bypass        : if True then a bypass will be added
        name              : name of module e.g. fire123

    Returns:
        x                 : returns a fire module
    """

    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze = tf.keras.layers.Conv2D(
        nb_squeeze_filter,
        (1, 1),
        activation="relu",
        padding="same",
        name="%s_squeeze" % name,
    )(x)
    expand_1x1 = tf.keras.layers.Conv2D(
        nb_expand_filter,
        (1, 1),
        activation="relu",
        padding="same",
        name="%s_expand_1x1" % name,
    )(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(
        nb_expand_filter,
        (3, 3),
        activation="relu",
        padding="same",
        name="%s_expand_3x3" % name,
    )(squeeze)

    axis = get_axis()
    x_ret = tf.keras.layers.Concatenate(axis=axis, name="%s_concatenate" % name)(
        [expand_1x1, expand_3x3]
    )

    if use_bypass:
        x_ret = tf.keras.layers.Add(name="%s_concatenate_bypass" % name)([x_ret, x])

    return x_ret


def get_axis():
    axis = -1 if K.image_data_format() == "channels_last" else 1
    return axis
