import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_name='configs', config_path= 'conf', version_base= None)
def my_app(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    my_app()