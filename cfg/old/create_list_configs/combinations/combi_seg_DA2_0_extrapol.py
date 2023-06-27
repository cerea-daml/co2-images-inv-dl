def return_dic_hyperparameters():
    """Create and return dictionnary of keys-values to combine."""

    base_cfg = "seg_PGIPW_DA2_0.cfg"
        
    
    dic_hyperparameters = {"orga.save.chain.bool": ["True"],
                           "data.input.xco2.noise.level": ["0.7"],
                           "model.name": ["Unet_efficientnetb0"],
                           "data.directory.name": ["LS_pPGIBJB"],
                          }
# , "LS_pPGIPW", "LS_pPGPP"
    return [base_cfg, dic_hyperparameters]