# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from tensorflow import keras
from wandb.keras import WandbCallback

import wandb



def get_modelcheckpoint(get: bool, cbs: list, filepath="w_best.h5") -> list:
    """Add modelcheckpoint to callbacks list if get."""
    if get:
        modelcheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            verbose=1,
        )
        cbs.append(modelcheckpoint_cb)
    else:
        pass
    return cbs


def get_lrscheduler(get: bool, cbs: list) -> list:
    """Add reducelronplateau to callbacks list if get."""
    if get:
        reducelronplateau_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            verbose=0,
            min_delta=5e-3,
            cooldown=0,
            min_lr=5e-5,
        )
        cbs.append(reducelronplateau_cb)
    else:
        pass
    return cbs


def get_earlystopping(get: bool, cbs: list) -> list:
    """Add earlystopping to callbacks list if get."""
    if get:
        earlystopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=5e-4,
            patience=50,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        cbs.append(earlystopping_cb)
    else:
        pass
    return cbs



