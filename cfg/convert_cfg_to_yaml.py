import os
from treeconfigparser import TreeConfigParser
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
initialize(config_path=".", job_name="job_name", version_base=None)

def get_dotlist_from_dict(d):
    """Get dot list for yaml compose from dict."""
    dot_list = []
    for k,v in d.items():
        dot_list.append(k +"=" + str(v))
    return dot_list

def get_myyaml_from_mycfg(cfg_tree: TreeConfigParser):
    """Get yaml coco2-cnn file from cfg coco2-cnn file."""
    
    if cfg_tree.get_stringlist("data.input.supps.list") == None or cfg_tree.get_stringlist("data.input.supps.list") == []:
        chan_1_type = "None"
    else:
        print("chan_1_type", chan_1_type)
        chan_1_type = cfg_tree.get_stringlist("data.input.supps.list")[0]
        
    if cfg_tree.get("data.output.label.weight.curve") == None:
        weighting_curve = "linear"
    else:
        weighting_curve = cfg_tree.get("data.output.label.weight.curve")
    
    dict_for_yaml = {"data.split.type": cfg_tree.get("data.tv_split"),
    "data.split.train.ratio": cfg_tree.get_float("data.training_ratio"),
    "data.input.chan_0.type": "xco2",
    "data.input.chan_1.type": chan_1_type,
    "data.input.chan_2.type": "None",
    "data.output.curve": weighting_curve,
    "data.output.min_w": cfg_tree.get_float("data.output.label.weight.min"),
    "data.output.max_w": cfg_tree.get_float("data.output.label.weight.max"),
    "data.output.param_curve": 1,
    "data.path.directory": cfg_tree.get("data.directory.main"),
    "data.path.train.name": cfg_tree.get("data.directory.name"),
    "data.path.train.nc": cfg_tree.get("2d_train_valid_dataset"),
    "data.path.valid.name": cfg_tree.get("data.directory.name"),
    "data.path.valid.nc": cfg_tree.get("2d_train_valid_dataset"),
    "data.path.test.name": cfg_tree.get("data.directory.name"),
    "data.path.test.nc": cfg_tree.get("2d_train_valid_dataset"),
    "data.training.batch_size": cfg_tree.get_int("model.batch_size"),
    "data.training.learning_rate": cfg_tree.get_float("model.lr.value"),
    "data.training.max_epochs": cfg_tree.get_int("model.epochs.number"),
    "data.training.init_weights": cfg_tree.get("model.init"),
    "data.training.optimiser": cfg_tree.get("model.optimiser"),
    "data.callbacks.model_checkpoint.__target__": cfg_tree.get("model.callback.modelcheckpoint"),
    "data.callbacks.learning_rate_monitor.__target__": cfg_tree.get("model.callback.reducelronplateau"),
    "data.callbacks.learning_rate_monitor.factor": 0.5,
    "data.callbacks.learning_rate_monitor.patience": 20,
    "data.callbacks.learning_rate_monitor.min_delta": 0.005,
    "data.callbacks.learning_rate_monitor.min_lr": 5E-5,
    "data.callbacks.learning_rate_monitor.cooldown": 0,
    "data.callbacks.wandb.__target__": cfg_tree.get("model.callback.wandb"),
    "data.dir_res": cfg_tree.get("orga.save.directory"),
    "data.exp_name": cfg_tree.get("orga.save.folder"),
    "data.seed": 42,
    "data.sweep": False,
    "data.model.type": cfg_tree.get("data.output.label.choice"),
    "data.model.name": cfg_tree.get("model.name"),
    "data.model.loss_func": cfg_tree.get("model.loss"),
    "data.model.dropout_rate": cfg_tree.get("model.dropout_rate"),
    "data.augmentations.rot.range": cfg_tree.get("data.input.aug.rot.range"),
    "data.augmentations.shift.range": cfg_tree.get("data.input.aug.shift.range"),
    "data.augmentations.flip.bool": cfg_tree.get("data.input.aug.flip.bool"),
    "data.augmentations.shear.range": cfg_tree.get("data.input.aug.shear.range"),
    "data.augmentations.zoom.range": cfg_tree.get("data.input.aug.zoom.range")}
    
    conf = OmegaConf.from_dotlist(get_dotlist_from_dict(dict_for_yaml))
    return conf

def save_myyaml_from_mycfg(dir_cfg):
    """Save yaml coco2-cnn file at directory of cfg coco2-cnn file."""
    cfg_file = os.path.join(dir_cfg, "config.cfg")
    cfg_tree = TreeConfigParser()
    cfg_tree.readfiles(cfg_file)
    cfg_yaml = get_myyaml_from_mycfg(cfg_tree)
    OmegaConf.save(cfg_yaml, os.path.join(dir_cfg, "config.yaml"))
    
    
