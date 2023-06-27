# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import tensorflow as tf


def linear_regressor(input_shape: list):
    """Linear regressor."""
    core_model = tf.keras.models.Sequential()
    core_model.add(tf.keras.Input(shape=input_shape))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.Flatten())
    core_model.add(tf.keras.layers.Dense(16))
    return core_model


def essential_regressor(input_shape: list):
    """Essential regressor."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    core_model = tf.keras.Model(inputs, x)
    return core_model


def essential_regressor_2(input_shape: list):
    """Essential regressor 2."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    core_model = tf.keras.Model(inputs, x)
    return core_model


def essential_regressor_3(input_shape: list):
    """Essential regressor 3."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    core_model = tf.keras.Model(inputs, x)
    return core_model
