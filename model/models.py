from model.optfusion_search import OptFusion_search
from optfusion.model.optfusion_train import OptFusion_train

def get_model(model_config, data_config):
    if model_config["model"] == "optfusion_search":
        model = OptFusion_search(data_config, model_config)
    elif model_config["model"] == "optfusion_train":
        model = OptFusion_train(data_config, model_config)
    else:
        raise ValueError("Invalid model type: {}".format(model_config["model"]))
    return model
