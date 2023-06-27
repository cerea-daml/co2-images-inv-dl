# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import pickle
import shutil
import sys

import joblib
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing


class Saver:
    """Saver of all results relevant to CNN model training experience."""

    def __init__(self):
        """Prepare directory to store results of the experiments."""

    def save_model_and_weights(self, model: tf.keras.Model):
        """Save model and weights using keras built_in functions."""
        model.save("w_last.h5")

    def save_data_shuffle_indices(self, ds_indices: dict):
        """Save shuffle indices."""
        with open("tv_inds.pkl", "wb") as f:
            pickle.dump(ds_indices, f)

    def save_input_scaler(self, scaler: preprocessing.StandardScaler):
        """Save input data sklearn scaler."""
        joblib.dump(scaler, "scaler.save")
