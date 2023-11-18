import hydra
from omegaconf import OmegaConf, DictConfig


# @hydra.main(config_name='configs', config_path= 'conf', version_base= None)
# # def my_app(cfg: DictConfig):
# #     config = cfg
# #     config = OmegaConf.to_yaml(cfg)
# #     return config

@hydra.main(config_name='configs', config_path= 'conf', version_base= None)
def main(cfg: OmegaConf):
    print(cfg.dataset.a_test_patient)


if __name__ == "__main__":
    main()

