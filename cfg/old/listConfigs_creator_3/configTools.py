#!/usr/bin/env python
#*--------------------------------------------------
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

from importeur import *
import itertools

#------------------------------------------------------------------------
# createDirSmashingOldOne
def createDirSmashingOldOne (directory):

    if os.path.exists (directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

#------------------------------------------------------------------------------------
# delete_value_corresponding_to_stringTarget_in_filename 
def delete_value_corresponding_to_stringTarget_in_filename (filename, stringTarget):
    
    with open(filename, 'r') as file:
        data = file.readlines()

    for i in range(len(data)):
        if (data[i].find(stringTarget) != -1):
            data[i] = stringTarget + ' = \n'

    with open(filename, 'w') as file:
        file.writelines (data)

    return 0

#------------------------------------------------------------------------------------
# replace_value_corresponding_to_stringTarget_in_filename
def replace_value_corresponding_to_stringTarget_in_filename (filename, stringTarget, newvalue) :
    
    with open(filename, 'r') as file:
        data = file.readlines()

    for i in range(len(data)):
        if (data[i].find(stringTarget) != -1):
            data[i] = stringTarget + ' = %s\n'%newvalue

    with open(filename, 'w') as file:
        file.writelines (data)

    return 0

#------------------------------------------------------------------------------------
# build_list_of_config_folders
def build_list_of_config_folders (originalFile_name, list_variables, list_combinations, chaintest, temporary_config):
    
    builtFile_name      = "config.cfg"
    builtFile_path      = temporary_config + '/' + builtFile_name
    number_variables    = len(list_variables)

    number_combi        = len(list_combinations)
    
    os.system ('rm -rf ' + temporary_config + '/*')
    os.system ('cp %s'%originalFile_name + ' ' + temporary_config + '/%s'%builtFile_name)
    print ('cp %s'%originalFile_name + ' ' + temporary_config + '/%s'%builtFile_name)

    # remove dans base config files values after variables of interest
    for i in range (number_variables):
        stringTarget = list_variables[i]
        delete_value_corresponding_to_stringTarget_in_filename (builtFile_path, stringTarget) 

    # remove save folder 
    stringTarget = "orga.save.folder"
    delete_value_corresponding_to_stringTarget_in_filename (builtFile_path, stringTarget)

    if number_combi > 1000:
        sys.exit()

    # for each combination, create a copy,   
    for i in range (number_combi):
   
        print ("create combination", i)
        
        # create folder for combination i
        current_directory = temporary_config + '/config%s/'%i 
        if not os.path.exists (current_directory):
            os.makedirs (current_directory)

        # copy config file in folder combi i
        os.system ('cp ' + temporary_config + '/%s'%builtFile_name + ' ' + temporary_config + '/config%s/'%i)
        
        # open built config file
        current_builtFile = current_directory + builtFile_name 
        with open(current_builtFile, 'r') as file:
            data = file.readlines()
 
        # change variables of the config file in folder combi i
        for k in range (number_variables):
            index_var       = data.index (list_variables[k] + ' = \n') 
            data[index_var] = list_variables[k] + ' = ' + str (list_combinations[i][k]) + '\n' 
 
        # change name of config.cfg in combi i  
        indexvar_file       = data.index ('orga.save.folder'  + ' = \n')
        data[indexvar_file] = 'orga.save.folder = ' + chaintest + '/test%s'%i 
        data[indexvar_file] = data[indexvar_file] + '\n'
   
        # write to file
        with open (current_builtFile, 'w') as file:
            file.writelines (data)

#------------------------------------------------------------------------------------
# create_clean_parameter_names_list
def create_clean_parameter_names_list (list_all_variables):
    
    list_clean_variables_names      = list() 
    
    for i in range (len(list_all_variables)) :

        if list_all_variables[i][0] == "model.name":
            list_clean_variables_names.append ("model.name")

        elif list_all_variables[i][0] == "model.optimiser":
            list_clean_variables_names.append ("optimiser")

        elif list_all_variables[i][0] == "model.learning_rate":
            list_clean_variables_names.append ("learning_rate")

        elif list_all_variables[i][0] == "model.batch_size":
            list_clean_variables_names.append ("batch_size")

        elif list_all_variables[i][0] == "data.directory.name":
            list_clean_variables_names.append ("dataset")

        elif list_all_variables[i][0] == "data.input.time":
            list_clean_variables_names.append ("input.time")

        elif list_all_variables[i][0] == "data.input.winds.format":
            list_clean_variables_names.append ("winds.format")

        elif list_all_variables[i][0] == "data.input.winds.fields.number.current":
            list_clean_variables_names.append ("N_wind_fields")

        elif list_all_variables[i][0] == "data.input.dynamics.format":
            list_clean_variables_names.append ("dynamic.format")

        elif list_all_variables[i][0] == "data.input.dynamics.fields.number.current":
            list_clean_variables_names.append ("N_dynamic_fields")

        else:
            list_clean_variables_names.append (list_all_variables[i][0][-15:])

    return list_clean_variables_names   

#------------------------------------------------------------------------------------


