#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import numpy as np
from tensorflow import keras
import tensorflow as tf
from dataclasses import dataclass


# --------------------------------------------------------
# standard_CNN
def standard_CNN(input_shape, Nclasses):

    inputs = keras.layers.Input(input_shape)

    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(
        inputs
    )
    x = keras.layers.Dropout(0.1)(x)    
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(x)    
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)    
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(Nclasses)(x)
    outputs = keras.layers.LeakyReLU(alpha=0.3)(x)

    model = keras.Model([inputs], outputs)

    return model

# PReLU, elu, LeakyReLU(alpha=0.3)
# --------------------------------------------------------
# classicCNN
def classic_CNN(name, input_shape, Nclasses, init_weights):

    if not (
        name.startswith("EfficientNet")
        or name.startswith("ResNet")
        or name.startswith("DenseNet")
    ):
        print("Wrong name neural network")
        sys.exit()

    if init_weights == "random":
        init_weights = None

    # model name can be
    ## EfficientNet[B0, B1, B2, ..., B7]
    ## ResNet[50V2, 101V2, 152V2]
    ## DenseNet[121, 169, 201]
    model_to_call = getattr(keras.applications, name)
    base_model = model_to_call(
        include_top=False,
        weights=init_weights,
        input_shape=input_shape,
    )
    base_model.trainable = True

    # build model
    inputs = keras.layers.Input(shape=input_shape, name="input_layer")
    # base model
    x = base_model(inputs)
    # feature extraction
    x = keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    dx = keras.layers.Dropout(rate=0.5, name="top_dropout")(x)
    x = keras.layers.Dense(Nclasses, name="output_layer")(x)
    """
    outputs = keras.layers.Activation(
        activation="LeakyReLU", alpha dtype=tf.float32, name="activation_layer"
    )(x)
    """
    outputs = keras.layers.LeakyReLU(alpha=0.3, dtype=tf.float32, name="activation_layer")(x)

    model = keras.Model([inputs], outputs)

    return model


@dataclass
class Keras_reg_model_builder:
    name: str
    input_shape: list
    classes: int
    init_w: str = None
    
    def get_model(self):
        """Return keras regression model."""
        base_model = self.define_base()
        model = self.add_top(base_model)
        return model
        
    def define_base(self):
        """Define base of the model with keras API."""
        model_to_call = getattr(keras.applications, self.name)
        base_model = model_to_call(
            include_top=False,
            weights=self.init_w,
            input_shape=self.input_shape,
        )
        base_model.trainable = True
        return base_model
            
    def add_top(self, base_model):
        """Add top layers to keras regression model."""
        inputs = keras.layers.Input(shape=self.input_shape, name="input_layer")
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        dx = keras.layers.Dropout(rate=0.5, name="top_dropout")(x)
        x = keras.layers.Dense(self.classes, name="output_layer")(x)
        outputs = keras.layers.LeakyReLU(alpha=0.3, dtype=tf.float32, name="activation_layer")(x)
        model = keras.Model([inputs], outputs)
        return model
    


@dataclass
class Reg_model_builder:
    """Return appropriate loss function."""
    name: str
    input_shape: np.ndarray
    classes: int
    init_w: str = "random"
    
    def get_model(self):
        """Return regression model, keras or locals."""
        if (
            self.name.startswith("EfficientNet")
            or self.name.startswith("ResNet")
            or self.name.startswith("DenseNet")
        ):
            keras_builder = Keras_reg_model_builder(self.name, self.input_shape, self.classes, self.init_w)
            model = keras_builder.get_model()
            
        else:
            model_to_call = locals()[self.name]
            model = standard_CNN(self.input_shape, self.classes)
        
        return model
