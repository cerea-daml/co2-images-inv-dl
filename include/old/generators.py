#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import numpy as np
from treeconfigparser import TreeConfigParser
from tensorflow import keras

# __________________________________________________________
# Generator
class Generator:

    # ----------------------------------------------------
    # __initialiser__
    def __init__(self, config, **kwargs):

        # parameters neural network
        self.model_name = config.get("model.name")
        self.batch_size = config.get_int("model.batch_size")
   
        # basic generator
        self.generator = "multiFields"
        
        self.rotation_range = config.get_int("data.input.aug.rot.range")
        self.width_shift_range = config.get_float("data.input.aug.height.range")
        self.height_shift_range = config.get_float("data.input.aug.width.range")
        self.horizontal_flip = config.get_bool("data.input.aug.hori.bool")
        self.vertical_flip = config.get_bool("data.input.aug.vert.bool")
        self.shear_range = config.get_float("data.input.aug.shear.range")
        self.zoom_range = config.get_float("data.input.aug.zoom.range")


        # multiFields_vector_CNN
        if self.model_name.startswith("multiFields_vector_CNN"):
            self.generator = "multiFields_vector"

        self.createTrainDataGenerator()

    # ------------------------------------------------------------------------
    # createTrainDataGenerator
    def createTrainDataGenerator(self):

        data_gen_args = dict(            rotation_range = self.rotation_range,
            width_shift_range = self.width_shift_range,
            height_shift_range = self.height_shift_range,
            horizontal_flip = self.horizontal_flip,
            vertical_flip = self.vertical_flip,
            shear_range = self.shear_range,
            zoom_range = self.zoom_range)
        
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

        self.trainData_generator = keras.preprocessing.image.ImageDataGenerator(
            rotation_range = self.rotation_range,
            width_shift_range = self.width_shift_range,
            height_shift_range = self.height_shift_range,
            horizontal_flip = self.horizontal_flip,
            vertical_flip = self.vertical_flip,
            shear_range = self.shear_range,
            zoom_range = self.zoom_range
        )

    # ------------------------------------------------------------------------
    # flow_on_data
    def flow_on_data(self, x_train, y_train):

        if self.generator.startswith("multiFields"):
            return self.flow_on_multiFields(x_train, y_train)

        if self.generator.startswith("multiFields_vector"):
            return self.flow_on_multiFields_vector(x_train, y_train)

    # ------------------------------------------------------------------------
    # flow_on_multiFields
    def flow_on_multiFields(self, x_train, y_train):

        return self.trainData_generator.flow(
            x_train[0], y_train, batch_size=self.batch_size, seed=27, shuffle=False
        )

    # ------------------------------------------------------------------------
    # flow_on_multiFields_vector
    def flow_on_multiFields_vector(self, x_train, y_train):

        while True:
            field_train = x_train[0]
            vector_train = x_train[1]
            n_training_data = field_train.shape[0]

            list_permuted_idx = np.random.permutation(n_training_data)

            batches = self.trainData_generator.flow(
                field_train[list_permuted_idx],
                y_train[list_permuted_idx],
                batch_size=self.batch_size,
                shuffle=False,
            )
            idx0 = 0
            for batch in batches:
                idx1 = idx0 + batch[0].shape[0]
                yield [batch[0], vector_train[list_permuted_idx[idx0:idx1]]], batch[1]
                idx0 = idx1
                if idx1 >= n_training_data:
                    break


# __________________________________________________________
