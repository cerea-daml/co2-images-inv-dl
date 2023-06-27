# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras


def pixel_weighted_cross_entropy(
    y_true: tf.Tensor, y_pred: tf.Tensor, reduction: bool = True
):
    """Loss function for segmentation weighted.
    Loss for weighting each pixel loss with y_true[pixel] value."""
    y_bin_true = tf.cast(y_true > 0, y_true.dtype)
    loss_val = keras.losses.binary_crossentropy(y_bin_true, y_pred)
    weights = tf.where(y_true > 0, y_true, 1.0)
    loss_val = tf.convert_to_tensor(tf.squeeze(weights), float) * loss_val
    if reduction:
        return K.mean(loss_val)
    if not reduction:
        return loss_val


def extreme_weighted_msle(y_mean, y_min, y_max):
    """Loss function for regression.
    Add loss weight to low and high labels in order to avoid mean regression."""
    w_max = 2
    w_min = 0.5

    a_min_mean = (w_max - w_min) / (y_min - y_mean)
    b_min_mean = w_max - a_min_mean * y_min

    a_max_mean = (w_max - w_min) / (y_max - y_mean)
    b_max_mean = w_max - a_max_mean * y_max

    def loss(y_true, y_pred):
        if y_true < y_mean:
            weight = a_min_mean * y_true + b_min_mean
            return weight * tf.keras.losses.MeanSquaredLogarithmicError(y_true, y_pred)
        elif y_true > y_mean:
            weight = a_max_mean * y_true + b_max_mean
            return weight * tf.keras.losses.MeanSquaredLogarithmicError(y_true, y_pred)

    return loss


def double_relative_error(y_true, y_pred):
    """Return relative mean absolute error."""
    return tf.math.reduce_mean((y_true - y_pred) ** 2 / (y_true * y_pred))


def define_loss(name_loss):
    """Return appropriate loss function."""
    call_dict = {
        "pixel_weighted_cross_entropy": pixel_weighted_cross_entropy,
        "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
        "MeanSquaredLogarithmicError": tf.keras.losses.MeanSquaredLogarithmicError(),
        "MeanAbsolutePercentageError": tf.keras.losses.MeanAbsolutePercentageError(),
        "MeanSquaredError": tf.keras.losses.MeanSquaredError(),
        "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError(),
    }
    loss = call_dict[name_loss]
    return loss


def define_metrics(exp_purpose: str):
    """Return list of metric functions."""
    metrics = []
    if exp_purpose == "segmentation":
        metrics = []
    elif exp_purpose == "inversion":
        metrics = [tf.keras.losses.MeanAbsolutePercentageError(), tf.keras.losses.MeanAbsoluteError()]
    return metrics


def calculate_weighted_plume(
    plume: np.ndarray,
    min_w: float,
    max_w: float,
    curve: str = "linear",
    param_curve: float = 1,
):
    """Calculate a weighted plume given min_w, max_w, and a curve between the two."""

    N_data = plume.shape[0]
    y_min = np.repeat(
        [np.where(plume > 0, plume, np.max(plume)).min()], N_data
    ).reshape(N_data, 1, 1)
    y_max = np.quantile(plume, q=0.995, axis=(1, 2)).reshape(N_data, 1, 1)
    weight_min = np.repeat([min_w], N_data).reshape(N_data, 1, 1)
    weight_max = np.repeat([max_w], N_data).reshape(N_data, 1, 1)

    if curve == "linear":
        pente = (weight_max - weight_min) / (y_max - y_min)  # type: ignore
        b = weight_min - pente * y_min
        y_data = pente * plume + b * np.where(plume > 0, 1, 0)

    elif curve == "exponential":
        A_0 = (weight_min - weight_max) / (  # type: ignore
            np.exp(param_curve * y_min) - np.exp(param_curve * y_max)
        )
        b = (
            weight_min
            + weight_max
            - A_0 * (np.exp(param_curve * y_min) + np.exp(param_curve * y_max))
        ) / 2
        y_data = A_0 * np.exp(param_curve * plume) + b * np.where(plume > 0, 1, 0)
        y_data = np.where(plume > 0, y_data, 0)
    else:
        sys.exit()

    y_data = np.where(y_data < max_w, y_data, max_w)
    y_data = np.expand_dims(y_data, axis=-1)

    return y_data
