from omegaconf import DictConfig, OmegaConf
import hydra

class Model_training_manager:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def try_method(self):
        a = self.cfg.data.callbacks.model_checkpoint
        print(a)


@hydra.main(version_base=None, config_path=".", config_name="test")
def my_app(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    
    m = Model_training_manager(cfg)
    m.try_method()
    

if __name__ == "__main__":
    my_app()
