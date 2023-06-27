#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

from tensorflow import keras
import tensorflow as tf

# --------------------------------------------------------
# build_multiFields_CNN
def build_multiFields_CNN(input_shape, out_Nclasses):

    # input
    multi_fields_input = keras.Input(shape=input_shape, name="multi-fields")

    # core
    multi_fields_data = convo_basis_for_multi_fields_input(multi_fields_input)

    # output
    output_tensor = keras.layers.Dense(out_Nclasses, activation="sigmoid")(
        multi_fields_data
    )

    # model
    model = keras.Model([multi_fields_input], output_tensor)

    return model


# --------------------------------------------------------
# build_multiFields_vector_CNN
def build_multiFields_vector_CNN(fields_input_shape, vector_input_shape, out_Nclasses):

    # input fields + convo basis
    multi_fields_input = keras.Input(shape=fields_input_shape, name="multi-fields")
    multi_fields_data = convo_basis_for_multi_fields_input(multi_fields_input)

    # input vector + dense basis
    vector_input = keras.Input(shape=vector_input_shape, name="vector")
    vector_data = keras.layers.Dense(8, activation="relu")(vector_input)
    vector_data = keras.layers.Flatten()(vector_data)

    # decision
    concatenated_data = keras.layers.concatenate(
        [multi_fields_data, vector_data], axis=-1
    )
    concatenated_data = keras.layers.Dense(32, activation="relu")(concatenated_data)

    # output
    output_tensor = keras.layers.Dense(out_Nclasses, activation="sigmoid")(
        concatenated_data
    )

    # model
    model = keras.Model([multi_fields_input, vector_input], output_tensor)

    return model


# --------------------------------------------------------
# convo_basis_for_multi_fields_input
def convo_basis_for_multi_fields_input(data_input):

    data = keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", strides=(1, 1)
    )(data_input)
    data = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=(2, 2))(
        data
    )
    data = keras.layers.BatchNormalization()(data)

    data = keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", strides=(1, 1)
    )(data)
    data = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=(2, 2))(
        data
    )
    data = keras.layers.BatchNormalization()(data)

    data = keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation="relu", strides=(1, 1)
    )(data)
    data = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=(2, 2))(
        data
    )
    data = keras.layers.BatchNormalization()(data)

    data = keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", strides=(1, 1)
    )(data)
    data = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=(2, 2))(
        data
    )
    data = keras.layers.BatchNormalization()(data)

    data = keras.layers.Flatten()(data)
    data = keras.layers.Dropout(0.5)(data)

    return data


# --------------------------------------------------------
# build_multiFields_classicNN
def build_multiFields_classicNN(input_shape, name, init_weights, out_Nclasses):

    if not (
        name.startswith("EfficientNet")
        or name.startswith("ResNet")
        or name.startswith("DenseNet")
    ):
        print("Wrong name neural network")
        sys.exit()
    
    if init_weights == "random":
        init_weights = None
    elif init_weights == "imagenet":
        pass
    else:
        print("Wrong weights init")
        sys.exit()

    activation = "sigmoid"
    if out_Nclasses > 1:
        activation = "softmax"
    elif out_Nclasses == 1:
        activation = "sigmoid"
    else:
        print ("Wrong number of classes")
        sys.exit()

    # model name can be
    ## EfficientNetB0 (B1, B2, ..., B7)
    ## ResNet50V2 (101V2, 152V2)
    ## DenseNet121 (169, 201)
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
    x = keras.layers.Dropout(rate=0.5, name="top_dropout")(x)
    x = keras.layers.Dense(out_Nclasses, name="output_layer")(x)
    outputs = keras.layers.Activation(
        activation=activation, dtype=tf.float32, name="activation_layer"
    )(x)

    model = keras.Model(inputs, outputs)

    return model


# --------------------------------------------------------
