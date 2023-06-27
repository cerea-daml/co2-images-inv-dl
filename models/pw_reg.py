#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

from tensorflow import keras

#--------------------------------------------------------
# Unet_1
def Unet_1(input_shape, Nclasses):

    inputs = keras.layers.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###
    # Entry block
            
    x = keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, kernel_size=(3,3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, kernel_size=(3,3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x)

        # Project residual
        residual    = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x           = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, kernel_size=(3,3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, kernel_size=(3,3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, kernel_size=(1,1), padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(out_Nclasses, kernel_size=(3,3), activation="relu", padding="same")(x)

    # Define the model
    model = keras.Model([inputs], outputs)

    return model

#--------------------------------------------------------
# Unet_2
def Unet_2(input_shape, Nclasses):
    
    [Ny, Nx, Nchannels] = input_shape
    
    inputs = keras.layers.Input((Ny, Nx, Nchannels))

    c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = keras.layers.Dropout(0.1) (c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = keras.layers.MaxPooling2D((2, 2)) (c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = keras.layers.Dropout(0.1) (c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = keras.layers.MaxPooling2D((2, 2)) (c2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = keras.layers.Dropout(0.2) (c3)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = keras.layers.MaxPooling2D((2, 2)) (c3)

    c4 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = keras.layers.Dropout(0.2) (c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = keras.layers.Dropout(0.3) (c5)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = keras.layers.Dropout(0.2) (c6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = keras.layers.Dropout(0.2) (c7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = keras.layers.Dropout(0.1) (c8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = keras.layers.Dropout(0.1) (c9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = keras.layers.Conv2D(Nclasses, (1, 1), activation='relu') (c9)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model

#--------------------------------------------------------

