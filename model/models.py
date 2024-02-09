from model.autofusion_search import AutoFusion_search
from autofusion.model.autofusion_train import AutoFusion_train

def get_model(model_config, data_config):
    if model_config["model"] == "autofusion_search":
        model = AutoFusion_search(data_config, model_config)
    elif model_config["model"] == "autofusion_train":
        model = AutoFusion_train(data_config, model_config)
    else:
        raise ValueError("Invalid model type: {}".format(model_config["model"]))
    return model