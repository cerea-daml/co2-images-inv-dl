# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import shutil
import itertools
import datetime
import glob
from dataclasses import dataclass, field
from typing import List


def build_all_combinations(dic_hyperparameters: dict) -> list:
    """Build combinations of key-vals."""
    list_combi = [
        dict(zip(dic_hyperparameters.keys(), vals))
        for vals in itertools.product(*dic_hyperparameters.values())
    ]
    return list_combi


def make_dic_headers(n_cfg: int, key_exp: str) -> dict:
    """Return dictionary for filling values related to header-orga."""
    chaintest_dir = "_".join(["chaintest", key_exp])
    dic_headers = [
        {
            "orga.save.folder": os.path.join(chaintest_dir, "test%s" % idx),
            "orga.save.chain.name": chaintest_dir,
        }
        for idx in range(n_cfg)
    ]
    return dic_headers


@dataclass
class Path_manager:
    """Build directory tree structure to store cfg files."""

    base_cfg: str
    cfg_name: str
    key_exp: str
    n_cfg: int

    def __post_init__(self):
        self.base_cfg = os.path.join("/cerea_raid/users/dumontj/dev/coco2/dl/cfg", self.base_cfg)
        
        self.temporary_config = os.path.join(
            "/cerea_raid/users/dumontj/dev/coco2/dl/cfg",
            "temporaryconfig",
            self.key_exp,
        )
        self.cfg_dirs = [
            os.path.join(self.temporary_config, "config%s" % idx)
            for idx in range(self.n_cfg)
        ]
        self.cfg_files = [
            os.path.join(cfg_dir, self.cfg_name) for cfg_dir in self.cfg_dirs
        ]

    def build_temporaryconfig(self) -> None:
        """Build temporaryconfig to store list of configuration files."""
        if os.path.exists(self.temporary_config):
            for folder in glob.glob(self.temporary_config + "/*"):
                try:
                    shutil.rmtree(folder, ignore_errors=False, onerror=None)
                except:
                    print("Problem deleting folders.")
        if not os.path.exists(self.temporary_config):
            os.makedirs(self.temporary_config)

    def make_cfg_dirs(self) -> None:
        """Make directories for all configuration files and copy config file inside."""
        for cfg_dir, cfg_file in zip(self.cfg_dirs, self.cfg_files):
            if not os.path.exists(cfg_dir):
                os.makedirs(cfg_dir)
            shutil.copyfile(self.base_cfg, cfg_file)


@dataclass
class Cfg_modificator:
    """Modify cfg files according to dic hyperparameters."""

    cfg_files: List[str]
    dic_hyperparameters: List[dict]
    dic_headers: dict = field(default_factory=lambda: dic_headers)

    def delete_vals_in_file(self, file_name: str, list_keys: list) -> None:
        """Delete list of values corresponding to keys in a file."""
        with open(file_name, "r") as file:
            data = file.readlines()
        for key in list_keys:
            for i in range(len(data)):
                if data[i].find(key) != -1:
                    data[i] = key + " = \n"
        with open(file_name, "w") as file:
            file.writelines(data)

    def add_vals_in_file(self, file_name: str, dic: dict) -> None:
        """Add values corresponding to keys of dic in file."""
        with open(file_name, "r") as file:
            data = file.readlines()
        for key, val in dic.items():
            idx_val = data.index(key + " = \n")
            data[idx_val] = key + " = " + val + "\n"
        with open(file_name, "w") as file:
            file.writelines(data)

    def modify_file(self, cfg_file: str, dic: dict, dic_header: dict) -> None:
        """Modify specific file according to specific dictionnary."""
        self.delete_vals_in_file(cfg_file, dic.keys())
        self.delete_vals_in_file(cfg_file, dic_header.keys())

        self.add_vals_in_file(cfg_file, dic)
        self.add_vals_in_file(cfg_file, dic_header)

    def modify_cfg_files(self) -> None:
        """Modify cfg files according to dic hyperparameters."""
        for file, dic, dic_header in zip(
            self.cfg_files, self.dic_hyperparameters, self.dic_headers
        ):
            self.modify_file(file, dic, dic_header)
