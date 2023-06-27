# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))

# see https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py

from __future__ import absolute_import, division, print_function

import collections
import math
import os
import string

import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from six.moves import xrange

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
)
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        input_filters=32,
        output_filters=16,
        expand_ratio=1,
        id_skip=True,
        strides=[1, 1],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        input_filters=16,
        output_filters=24,
        expand_ratio=3,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=2,
        input_filters=24,
        output_filters=40,
        expand_ratio=3,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=2,
        input_filters=40,
        output_filters=80,
        expand_ratio=3,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=2,
        input_filters=80,
        output_filters=112,
        expand_ratio=4,
        id_skip=True,
        strides=[1, 1],
        se_ratio=0.25,
    ),
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        "distribution": "normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


def get_swish(**kwargs):
    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return tf.nn.swish(x)
        except AttributeError:
            pass

        return x * tf.sigmoid(x)

    return swish


def get_dropout(**kwargs):
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.
    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """

    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.shape(inputs)
            noise_shape = [
                symbolic_shape[axis] if shape is None else shape
                for axis, shape in enumerate(self.noise_shape)
            ]
            return tuple(noise_shape)

    return FixedDropout


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(
    inputs,
    block_args,
    activation,
    drop_rate=None,
    prefix="",
):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3
    # workaround over non working dropout with None in noise_shape in tf.keras
    Dropout = get_dropout()

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = tf.keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + "expand_conv",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "expand_bn")(
            x
        )
        x = tf.keras.layers.Activation(activation, name=prefix + "expand_activation")(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D(
        block_args.kernel_size,
        strides=block_args.strides,
        padding="same",
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix + "dwconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "bn")(x)
    x = tf.keras.layers.Activation(activation, name=prefix + "activation")(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(
            1, int(block_args.input_filters * block_args.se_ratio)
        )
        se_tensor = tf.keras.layers.GlobalAveragePooling2D(name=prefix + "se_squeeze")(
            x
        )

        target_shape = (1, 1, filters)
        se_tensor = tf.keras.layers.Reshape(target_shape, name=prefix + "se_reshape")(
            se_tensor
        )
        se_tensor = tf.keras.layers.Conv2D(
            num_reduced_filters,
            1,
            activation=activation,
            padding="same",
            use_bias=True,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + "se_reduce",
        )(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(
            filters,
            1,
            activation="sigmoid",
            padding="same",
            use_bias=True,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=prefix + "se_expand",
        )(se_tensor)
        x = tf.keras.layers.multiply([x, se_tensor], name=prefix + "se_excite")

    # Output phase
    x = tf.keras.layers.Conv2D(
        block_args.output_filters,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix + "project_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + "project_bn")(x)
    if (
        block_args.id_skip
        and all(s == 1 for s in block_args.strides)
        and block_args.input_filters == block_args.output_filters
    ):
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + "drop")(x)
        x = tf.keras.layers.add([x, inputs], name=prefix + "add")

    return x


def EfficientNet_constructor(
    width_coefficient,
    depth_coefficient,
    default_resolution=64,
    drop_connect_rate=0.2,
    depth_divisor=8,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    model_name="efficientnet",
    include_top=False,
    weights=None,
    input_shape=None,
    **kwargs
):

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_resolution,
        min_size=32,
        data_format="channels_last",
        require_flatten=include_top,
        weights=weights,
    )

    bn_axis = 3
    activation = get_swish(**kwargs)

    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        round_filters(32, width_coefficient, depth_divisor),
        3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="stem_conv",
    )(img_input)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = tf.keras.layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    num_blocks_total = sum(
        round_repeats(block_args.num_repeat, depth_coefficient)
        for block_args in blocks_args
    )
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(
                block_args.input_filters, width_coefficient, depth_divisor
            ),
            output_filters=round_filters(
                block_args.output_filters, width_coefficient, depth_divisor
            ),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient),
        )

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(
            x,
            block_args,
            activation=activation,
            drop_rate=drop_rate,
            prefix="block{}a_".format(idx + 1),
        )
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1]
            )
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = "block{}{}_".format(
                    idx + 1, string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(
                    x,
                    block_args,
                    activation=activation,
                    drop_rate=drop_rate,
                    prefix=block_prefix,
                )
                block_num += 1

    # Build top
    x = tf.keras.layers.Conv2D(
        round_filters(1280, width_coefficient, depth_divisor),
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="top_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
    x = tf.keras.layers.Activation(activation, name="top_activation")(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name=model_name)

    return model


def EfficientNet(
    include_top=False,
    scaling_coefficient=1.0,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1,
    dropout_rate=0.2,
    **kwargs
):
    return EfficientNet_constructor(
        width_coefficient=scaling_coefficient,
        depth_coefficient=scaling_coefficient,
        default_resolution=64,
        dropout_rate=dropout_rate,
        model_name="efficientnetb0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )
