def return_dic_hyperparameters():
    """Create and return dictionnary of keys-values to combine."""

    base_cfg = "seg_PGIPW_DA2_1.cfg"
    
    dic_hyperparameters = {"orga.save.chain.bool": ["True"],
                           "data.input.xco2.noise.level": ["0.7"],
                           "model.name": ["Unet_5", "Unet_efficientnetb1", "Unet_efficientnetb0"],
                          }

    return [base_cfg, dic_hyperparameters]