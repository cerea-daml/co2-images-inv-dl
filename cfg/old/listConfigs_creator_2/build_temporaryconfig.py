#!/usr/bin/env python

# *--------------------------------------------------
# * /cerea_raid/users/dumontj/dev/plumeDetection/models/config/listConfigsCreation_2
# *
# *--------------------------------------------------
# * author  : joffreydumont@hotmail.fr
# * created : 2021
# *--------------------------------------------------
# *
# * Implementation of build_temporaryconfig.py
# * TODO:
# *
#

from importeur import *
from configTools import build_list_of_config_folders, create_clean_parameter_names_list
from list_combinations import *

if __name__ == "__main__":

    # set directories
    base_configfile = "presence_base_config.cfg"
    number_config = "2"
    chaintest = "chaintest" + number_config
    temporary_config = "temporaryconfig" + number_config
    basic_dir = "/cerea_raid/users/dumontj/dev/plumeDetection/models/config/"

    # config
    config = TreeConfigParser()
    config.readfiles(basic_dir + base_configfile)
    directory_res = config.get("orga.save.directory") + "/" + chaintest
    os.system("cp list_combinations.py " + directory_res)

    # build temporary config
    os.system("mkdir " + temporary_config)
    list_variables_names = list()
    for i in range(len(list_all_variables)):
        list_variables_names.append(list_all_variables[i][0])

    list_combinations = build_list_combinations()
    build_list_of_config_folders(
        basic_dir + base_configfile,
        list_variables_names,
        list_combinations,
        chaintest,
        temporary_config,
    )

    # replace old temporary config with new temporary config
    os.system("rm -rf ../" + temporary_config)
    os.system("mv " + temporary_config + " ../")

    # write table of combinations
    list_clean_variables_names = create_clean_parameter_names_list(list_all_variables)
    ## write combinations
    with open("table_combinations.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        ### write the header
        writer.writerow(list_clean_variables_names)
        ### write values
        writer.writerows(list_combinations)
    os.system("cp table_combinations.csv " + directory_res)

# ------------------------------------------------------------------------------------
