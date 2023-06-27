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


def initiate_wb(cfg: DictConfig) -> None:
    """Initiate Weight and Biases."""

    if cfg.callbacks.wandb:
        config_wb = {
            "train": cfg.data.path.train.name,
            "valid": cfg.data.path.valid.name,
            "model": cfg.model.name, 
            "chan_0": cfg.data.input.chan_0,
            "chan_1": cfg.data.input.chan_1,
            "chan_2": cfg.data.input.chan_2,
            "chan_3": cfg.data.input.chan_3,
            "chan_4": cfg.data.input.chan_4,
            "p_scal_min": cfg.augmentations.plume_scaling_min,
            "p_scal_max": cfg.augmentations.plume_scaling_max,
        }

        if cfg.sweep:
            wandb.init(
                project=cfg.exp_name,
                config=config_wb,
                name=os.path.basename(os.getcwd()),
                settings=wandb.Settings(start_method="thread"),
            )
        else:
            wandb.init(
                project=cfg.model.type,
                config=config_wb,
                settings=wandb.Settings(start_method="thread"),
            )


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


def get_wandb(get: bool, cbs: list) -> list:
    """Add wandb to callbacks list if get."""
    if get:
        cbs.append(WandbCallback())
    else:
        pass
    return cbs


class ExtraValidation(tf.keras.callbacks.Callback):
    def __init__(self, extra_val_data):
        super(ExtraValidation, self).__init__()

        self.extra_val_data = extra_val_data

    def on_epoch_end(self, epoch, logs=None):
        (extra_val_data, extra_val_targets) = self.extra_val_data
        extra_val_loss = self.model.evaluate(
            extra_val_data, extra_val_targets, verbose=0
        )
        print("extra_val_loss:", extra_val_loss)
        if type(extra_val_loss) == list:
            wandb.log({"extra_val_loss": extra_val_loss[0]})
            for idx, metric in enumerate(extra_val_loss[1:]):
                wandb.log({f"extra_val_metric_{idx}": metric})

        else:
            wandb.log({"extra_val_loss": extra_val_loss})
