from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(config_path=".", job_name="job_name")
cfg = compose(config_name="config")
print(OmegaConf.to_yaml(cfg))
