# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
from dataclasses import dataclass, field

import tensorflow as tf

import models.Unet_backboned as Uback
from models.my_essential_Unet import Unet_4, Unet_5


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


def get_core_model(
    name: str,
    input_shape: list,
    classes: int = 1,
    dropout_rate: float = 0.2,
):
    """Get core for segmentation model."""

    if name.startswith("Unet_efficientnetb"):
        encoder_name = {
            "Unet_efficientnetb0": "EfficientNetB0",
            "Unet_efficientnetb1": "EfficientNetB1",
            "Unet_efficientnetb2": "EfficientNetB2",
            "Unet_efficientnetb3": "EfficientNetB3",
            "Unet_efficientnetb4": "EfficientNetB4",
            "Unet_efficientnetb5": "EfficientNetB5",
            "Unet_efficientnetb6": "EfficientNetB6",
        }[name]
        model = Uback.Unet(
            encoder_name,
            input_shape=input_shape,
            classes=classes,
            drop_encoder_rate=dropout_rate,
        )

    else:
        model_names = {"Unet_4": Unet_4, "Unet_5": Unet_5}
        model = model_names[name](input_shape, classes)

    return model


@dataclass
class Seg_model_builder:
    """Return appropriate segmentation model."""

    name: str = "Unet_efficientnetb0"
    input_shape: list = field(default_factory=lambda: [64, 64, 1])
    classes: int = 1
    n_layer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(axis=-1)
    noisy_chans: list = field(
        default_factory=lambda: [True, False, False, False, False]
    )
    dropout_rate: float = 0.2

    def get_model(self):
        """Return segmentation model, from local or segmentation_models (old)."""

        bottom_layers = get_preprocessing_layers(
            self.n_layer, self.input_shape[-1], self.noisy_chans
        )
        core_model = get_core_model(
            self.name,
            self.input_shape,
            self.classes,
            self.dropout_rate,
        )

        inputs = tf.keras.layers.Input(self.input_shape, name="input_layer")
        x = bottom_layers(inputs)
        outputs = core_model(x)

        model = tf.keras.Model(inputs, outputs)

        return model

    """
    model = ext_sm.Unet(backbone_name="efficientnetb1", 
                                encoder_weights=None, input_shape=self.input_shape)
    """
