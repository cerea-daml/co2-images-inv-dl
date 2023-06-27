# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.my_efficientnet import EfficientNet
from models.my_essential_inversors import (essential_regressor,
                                           essential_regressor_2,
                                           essential_regressor_3,
                                           linear_regressor)
from models.my_mobilenet import MobileNet
from models.my_shufflenet import ShuffleNet
from models.my_squeezenet import SqueezeNet


def get_preprocessing_layers(
    n_layer: tf.keras.layers.Normalization, n_chans: int, noisy_chans: list
):
    """Return preprocessing layers for regression model."""

    def preproc_layers(x):
        chans = [None] * n_chans
        for idx in range(n_chans):
            if noisy_chans[idx]:
                chans[idx] = tf.keras.layers.GaussianNoise(
                    stddev=0.7, name=f"noise_{idx}"
                )(x[:, :, :, idx : idx + 1])
            else:
                chans[idx] = tf.keras.layers.Layer()(x[:, :, :, idx : idx + 1])

        concatted = tf.keras.layers.Concatenate()(chans)
        x = n_layer(concatted)
        return x

    return preproc_layers


def get_top_layers(classes: int, choice_top: str = "linear"):
    """Return top layers for regression model."""

    def top_layers(x):
        if choice_top in [
            "efficientnet",
            "squeezenet",
            "nasnet",
            "mobilenet",
            "shufflenet",
        ]:
            x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
            x = tf.keras.layers.Dense(classes, name="regressor")(x)
            outputs = tf.keras.layers.LeakyReLU(
                alpha=0.3, dtype=tf.float32, name="regressor_activ"
            )(x)
        elif choice_top == "linear":
            outputs = tf.keras.layers.Dense(classes, name="regressor")(x)
        elif choice_top.startswith("essential"):
            x = tf.keras.layers.Dense(1)(x)
            outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        else:
            sys.exit()
        return outputs

    return top_layers


def get_core_model(
    name: str,
    input_shape: list,
    classes: int = 1,
    dropout_rate: float = 0.2,
    scaling_coefficient: float = 1,
):
    """Get core model for regression model."""
    if name == "efficientnet":
        core_model = EfficientNet(
            scaling_coefficient=0.5,
            input_shape=input_shape,
            classes=classes,
            dropout_rate=dropout_rate,
        )
    elif name == "linear":
        core_model = linear_regressor(input_shape)
    elif name == "essential":
        core_model = essential_regressor(input_shape)
    elif name == "essential_2":
        core_model = essential_regressor_2(input_shape)
    elif name == "essential_3":
        core_model = essential_regressor_3(input_shape)
    elif name == "squeezenet":
        core_model = SqueezeNet(input_shape, dropout_rate, compression=0.4)
    elif name == "mobilenet":
        core_model = MobileNet(input_shape, scaling_coeff=0.4)
    elif name == "shufflenet":
        core_model = ShuffleNet(input_shape, scaling_coefficient=0.75)
    else:
        sys.exit()

    return core_model


@dataclass
class Reg_model_builder:
    """Return appropriate regression model."""

    name: str = "linear"
    input_shape: list = field(default_factory=lambda: [64, 64, 3])
    classes: int = 1
    n_layer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(axis=-1)
    noisy_chans: list = field(
        default_factory=lambda: [True, False, False, False, False]
    )
    dropout_rate: float = 0.2
    scaling_coefficient: float = 1

    def get_model(self):
        """Return regression model, keras or locals."""

        bottom_layers = get_preprocessing_layers(
            self.n_layer, self.input_shape[-1], self.noisy_chans
        )
        core_model = get_core_model(
            self.name,
            self.input_shape,
            self.classes,
            self.dropout_rate,
            self.scaling_coefficient,
        )
        top_layers = get_top_layers(self.classes, self.name)

        inputs = tf.keras.layers.Input(self.input_shape, name="input_layer")
        x = bottom_layers(inputs)
        x = core_model(x)
        outputs = top_layers(x)

        model = tf.keras.Model(inputs, outputs)

        return model
