# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# -------------------------------------Conv2D---------------------------------------

from functools import partial

import tensorflow as tf


def Unet_4(input_shape, Nclasses):
    """Unet ~ 2millions parameters - four phases."""

    [Ny, Nx, Nchannels] = input_shape

    inputs = tf.keras.layers.Input((Ny, Nx, Nchannels))

    c1 = tf.keras.layers.Conv2D(
        16, (3, 3), padding="same", activation="elu", kernel_initializer="he_normal"
    )(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(
        c5
    )
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(
        16, (3, 3), activation="elu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = tf.keras.layers.Conv2D(Nclasses, (1, 1), activation="sigmoid")(c9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model


def Unet_5(input_shape, Nclasses, dropout_rate=0.1):
    """Unet ~ 5millions parameters - five phases."""

    conv2d_elu_he = partial(
        tf.keras.layers.Conv2D,
        activation="elu",
        kernel_initializer="he_normal",
        padding="same",
    )

    [Ny, Nx, Nchannels] = input_shape
    N_chan_i = 16

    inputs = tf.keras.layers.Input((Ny, Nx, Nchannels))

    """Downsampling with downsampler and convolutions."""
    d_c0 = conv2d_elu_he(N_chan_i, (3, 3))(inputs)
    d_c0 = tf.keras.layers.Dropout(dropout_rate)(d_c0)
    d_c0 = conv2d_elu_he(N_chan_i, (3, 3))(d_c0)
    d_c0 = tf.keras.layers.BatchNormalization()(d_c0)

    d_d0 = tf.keras.layers.MaxPooling2D((2, 2))(d_c0)

    d_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(d_d0)
    d_c1 = tf.keras.layers.Dropout(dropout_rate)(d_c1)
    d_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(d_c1)
    d_c1 = tf.keras.layers.BatchNormalization()(d_c1)

    d_d1 = tf.keras.layers.MaxPooling2D((2, 2))(d_c1)

    d_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(d_d1)
    d_c2 = tf.keras.layers.Dropout(dropout_rate * 2)(d_c2)
    d_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(d_c2)
    d_c2 = tf.keras.layers.BatchNormalization()(d_c2)

    d_d2 = tf.keras.layers.MaxPooling2D((2, 2))(d_c2)

    d_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(d_d2)
    d_c3 = tf.keras.layers.Dropout(dropout_rate * 2)(d_c3)
    d_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(d_c3)
    d_c3 = tf.keras.layers.BatchNormalization()(d_c3)

    d_d3 = tf.keras.layers.MaxPooling2D((2, 2))(d_c3)

    d_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_d3)
    d_c4 = tf.keras.layers.Dropout(dropout_rate * 3)(d_c4)
    d_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_c4)
    d_c4 = tf.keras.layers.BatchNormalization()(d_c4)

    d_d4 = tf.keras.layers.MaxPooling2D((2, 2))(d_c4)

    """Mid-part."""
    m = conv2d_elu_he(N_chan_i * 16, (3, 3))(d_d4)
    m = tf.keras.layers.Dropout(dropout_rate * 3)(m)
    m = conv2d_elu_he(N_chan_i * 16, (3, 3))(m)
    m = tf.keras.layers.BatchNormalization()(m)

    """Upsampling with upsampler, residual, and convolutions."""
    u_u4 = tf.keras.layers.Conv2DTranspose(
        N_chan_i * 16, (2, 2), strides=(2, 2), padding="same"
    )(m)
    u_r4 = tf.keras.layers.concatenate([u_u4, d_c4])

    u_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(u_r4)
    u_c4 = tf.keras.layers.Dropout(dropout_rate * 3)(u_c4)
    u_c4 = conv2d_elu_he(N_chan_i * 16, (3, 3))(u_c4)
    u_c4 = tf.keras.layers.BatchNormalization()(u_c4)

    u_u3 = tf.keras.layers.Conv2DTranspose(
        N_chan_i * 8, (2, 2), strides=(2, 2), padding="same"
    )(u_c4)
    u_r3 = tf.keras.layers.concatenate([u_u3, d_c3])

    u_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(u_r3)
    u_c3 = tf.keras.layers.Dropout(dropout_rate * 2)(u_c3)
    u_c3 = conv2d_elu_he(N_chan_i * 8, (3, 3))(u_c3)
    u_c3 = tf.keras.layers.BatchNormalization()(u_c3)

    u_u2 = tf.keras.layers.Conv2DTranspose(
        N_chan_i * 4, (2, 2), strides=(2, 2), padding="same"
    )(u_c3)
    u_r2 = tf.keras.layers.concatenate([u_u2, d_c2])

    u_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(u_r2)
    u_c2 = tf.keras.layers.Dropout(dropout_rate * 2)(u_c2)
    u_c2 = conv2d_elu_he(N_chan_i * 4, (3, 3))(u_c2)
    u_c2 = tf.keras.layers.BatchNormalization()(u_c2)

    u_u1 = tf.keras.layers.Conv2DTranspose(
        N_chan_i * 2, (2, 2), strides=(2, 2), padding="same"
    )(u_c2)
    u_r1 = tf.keras.layers.concatenate([u_u1, d_c1])

    u_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(u_r1)
    u_c1 = tf.keras.layers.Dropout(dropout_rate)(u_c1)
    u_c1 = conv2d_elu_he(N_chan_i * 2, (3, 3))(u_c1)
    u_c1 = tf.keras.layers.BatchNormalization()(u_c1)

    u_u0 = tf.keras.layers.Conv2DTranspose(
        N_chan_i, (2, 2), strides=(2, 2), padding="same"
    )(u_c1)
    u_r0 = tf.keras.layers.concatenate([u_u0, d_c0])

    u_c0 = conv2d_elu_he(N_chan_i, (3, 3))(u_r0)
    u_c0 = tf.keras.layers.Dropout(dropout_rate)(u_c0)
    u_c0 = conv2d_elu_he(N_chan_i, (3, 3))(u_c0)
    u_c0 = tf.keras.layers.BatchNormalization()(u_c0)

    outputs = tf.keras.layers.Conv2D(Nclasses, (1, 1), activation="sigmoid")(u_c0)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model
