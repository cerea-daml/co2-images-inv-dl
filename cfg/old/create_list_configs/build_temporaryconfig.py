# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import shutil
import sys
import glob
import importlib
import datetime
import bash

import config_tools

sys.path.append("combinations")

if __name__ == "__main__":

    in_command = bash.config_file_names_from_command()
    experience = os.path.basename(in_command[0]).replace(".py", "")

    key_exp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    combi = importlib.import_module(experience)
    base_cfg, dic_hyperparameters = combi.return_dic_hyperparameters()
    list_combi = config_tools.build_all_combinations(dic_hyperparameters)

    path_man = config_tools.Path_manager(
        base_cfg, "config.cfg", key_exp, len(list_combi)
    )
    path_man.build_temporaryconfig()
    path_man.make_cfg_dirs()

    cfg_mod = config_tools.Cfg_modificator(
        path_man.cfg_files,
        list_combi,
        config_tools.make_dic_headers(len(list_combi), key_exp),
    )
    cfg_mod.modify_cfg_files()
