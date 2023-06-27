#!/usr/bin/env python
# *--------------------------------------------------
# * dev/plumeDetection/models/config/listConfigsCreation/
# *
# *--------------------------------------------------
# * author  : joffreydumont@hotmail.fr
# * created : 2021
# *--------------------------------------------------
# *
# * Implementation
# * TODO:
# *
#

import os
import shutil
import itertools

# ------------------------------------------------------------------------------------
# build_all_combinations
def build_all_combinations(list_all_variables):

    list_variables_values = [item[1:] for item in list_all_variables]
    list_combinations = list(itertools.product(*list_variables_values))

    return list_combinations


# ------------------------------------------------------------------------------------
# remove_impossible_combinations
def remove_impossible_combinations(
    list_all_variables, list_combinations, list_together_values_to_destroy
):

    dic_nameVar_idxVar = {}
    values = range(len(list_all_variables))
    keys = [item[0] for item in list_all_variables]
    for i in values:
        dic_nameVar_idxVar[keys[i]] = values[i]

    clean_list_combinations = []

    # append only possible combinations
    initial_number_combinations = len(list_combinations)
    for combination in range(initial_number_combinations):
        destroy = False
        for index_destruction in range(len(list_together_values_to_destroy)):
            N_together_values = len(list_together_values_to_destroy[index_destruction])
            if all(
                list_together_values_to_destroy[index_destruction][i][1]
                == list_combinations[combination][
                    dic_nameVar_idxVar[list_together_values_to_destroy[index_destruction][i][0]]
                ]
                for i in range(N_together_values)
            ):
                destroy = True

        if not destroy:
            clean_list_combinations.append(list_combinations[combination])

    return clean_list_combinations

# ------------------------------------------------------------------------------------
# delete_value_corresponding_to_stringTarget_in_filename
def delete_value_corresponding_to_stringTarget_in_filename(filename, stringTarget):

    with open(filename, "r") as file:
        data = file.readlines()

    for i in range(len(data)):
        if data[i].find(stringTarget) != -1:
            data[i] = stringTarget + " = \n"

    with open(filename, "w") as file:
        file.writelines(data)

    return 0


# ------------------------------------------------------------------------------------
# build_list_of_config_folders
def build_list_of_config_folders(
    original_base_config_file, list_variables, list_combinations, chaintest, temporary_config
):

    # copy base config
    config_file_name = "config.cfg"
    copy_base_config_file = os.path.join(temporary_config, config_file_name)
    shutil.copyfile(original_base_config_file, copy_base_config_file)

    # remove in copy base config 
    ## values after variables of interest
    number_variables = len(list_variables)
    for i in range(number_variables):
        stringTarget = list_variables[i]
        delete_value_corresponding_to_stringTarget_in_filename(
            copy_base_config_file, stringTarget
        )
    ## folder for saving
    stringTarget = "orga.save.folder"
    delete_value_corresponding_to_stringTarget_in_filename(copy_base_config_file, stringTarget)

    ## folder name chain
    stringTarget = "orga.save.chain.name"
    delete_value_corresponding_to_stringTarget_in_filename(copy_base_config_file, stringTarget)


    # for each combination, create a copy of copy base config and modify it
    number_combi = len(list_combinations)
    for i in range(number_combi):

        ## create folder for combination i
        dir_i = os.path.join(temporary_config, "config%s"%i)
        
        if not os.path.exists(dir_i):
            os.makedirs(dir_i)

        config_file_i = os.path.join(dir_i, "config.cfg")
        shutil.copyfile(copy_base_config_file, config_file_i)

        ## open built config file
        with open(config_file_i, "r") as file:
            data_i = file.readlines()

        ## change variables of the config file in folder combi i
        for k in range(number_variables):
            index_var = data_i.index(list_variables[k] + " = \n")
            data_i[index_var] = (
                list_variables[k] + " = " + str(list_combinations[i][k]) + "\n"
            )

        ## change name of config.cfg in combi i
        indexvar_file = data_i.index("orga.save.folder" + " = \n")
        data_i[indexvar_file] = "orga.save.folder = " + chaintest + "/test%s" % i
        data_i[indexvar_file] = data_i[indexvar_file] + "\n"

        ## change chain name of config.cfg in combi i
        indexvar_file = data_i.index("orga.save.chain.name" + " = \n")
        data_i[indexvar_file] = "orga.save.chain.name = " + chaintest 
        data_i[indexvar_file] = data_i[indexvar_file] + "\n"


        ## write to file
        with open(config_file_i, "w") as file:
            file.writelines(data_i)

# ------------------------------------------------------------------------------------
# create_clean_parameter_names_list
def create_clean_parameter_names_list(list_all_variables):

    list_clean_variables_names = list()

    for i in range(len(list_all_variables)):

        if list_all_variables[i][0] == "model.name":
            list_clean_variables_names.append("model.name")

        elif list_all_variables[i][0] == "model.optimiser":
            list_clean_variables_names.append("optimiser")

        elif list_all_variables[i][0] == "model.learning_rate":
            list_clean_variables_names.append("learning_rate")

        elif list_all_variables[i][0] == "model.batch_size":
            list_clean_variables_names.append("batch_size")

        elif list_all_variables[i][0] == "data.directory.name":
            list_clean_variables_names.append("dataset")

        elif list_all_variables[i][0] == "data.input.time":
            list_clean_variables_names.append("input.time")

        elif list_all_variables[i][0] == "data.input.winds.format":
            list_clean_variables_names.append("winds.format")

        elif list_all_variables[i][0] == "data.input.winds.fields.number.current":
            list_clean_variables_names.append("N_wind_fields")

        elif list_all_variables[i][0] == "data.input.dynamics.format":
            list_clean_variables_names.append("dynamic.format")

        elif list_all_variables[i][0] == "data.input.dynamics.fields.number.current":
            list_clean_variables_names.append("N_dynamic_fields")

        else:
            list_clean_variables_names.append(list_all_variables[i][0][-15:])

    return list_clean_variables_names


# ------------------------------------------------------------------------------------
