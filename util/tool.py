import os
import torch
import random
import numpy as np
import tensorflow as tf
from dataloader.tfloader import CriteoLoader, AvazuLoader, KDD12loader
from train.optimizers import Adam_Dev, Lamb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(gpu):
    device= torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpu))
    print("training on", device)
    return device

def get_data_loader(dataset):
    if dataset == "criteo_2":
        data_config = {
            "dataset": "criteo_2",
            "data_path": "./criteo_new/threshold_2/",
            "feature_num": 6780382,
            "field_num": 39,
            "sample_nums": {
                "train": 36672494,
                "valid": 4584062,
                "test": 4584061,
            },
        }
        data_loader = CriteoLoader(data_config["data_path"])
    
    elif dataset == "avazu_2":
        data_config = {
            "dataset": "avazu_2",
            "data_path": "./avazu_new/threshold_2/",
            "feature_num": 4428293,
            "field_num": 24,
            "sample_nums": {
                "train": 32343174,
                "valid": 4042896,
                "test": 4042897,
            },
        }
        data_loader = AvazuLoader(data_config["data_path"])
    
    elif dataset == "kdd_2":
        data_config = {
            "dataset": "kdd_2",
            "data_path": "./kdd_new/threshold_2/",
            "feature_num": 35970485,
            "field_num": 11,
            "sample_nums": {
                "train": 119711284,
                "valid": 14963910,
                "test": 14963911,
            },
        }
        data_loader = KDD12loader(data_config["data_path"])
    else:
        raise ValueError("not support dataset")
    return data_loader, data_config

def get_optimizer(network, params):
    embedding_params = []
    nn_params = []
    arch_params = []
    for name, param in network.named_parameters():
        if name.startswith("embedding"):
            embedding_params.append(param)
            print("embedding name; param_size:", name, param.size())
        elif name.startswith("connection_params_init") or name.startswith("fusion_params_init_list"):
            arch_params.append(param)
            print("arch_params name; param_size:", name, param.size())
        else:
            nn_params.append(param)
            print("nn_params name; param_size:", name, param.size())
        # print("name; param_size:", name, param.size()) #debug

    embedding_group = {
        'params': embedding_params,
        'weight_decay': params["l2_emb"],
        'lr': params['lr_emb'],
    }
    nn_group = {
        'params': nn_params,
        'weight_decay': params["l2_nn"],
        'lr':params['lr_nn'],
    }

    optimizer = torch.optim.Adam([embedding_group, nn_group])

    return optimizer

def get_optimizer_train(network, params):
    embedding_params = []
    nn_params = []
    for name, param in network.named_parameters():
        if name.startswith("embedding"):
            embedding_params.append(param)
            print("embedding name; param_size:", name, param.size())
        else:
            nn_params.append(param)
            print("nn_params name; param_size:", name, param.size())
        # print("name; param_size:", name, param.size()) #debug

    embedding_group = {
        'params': embedding_params,
        'weight_decay': params["l2_emb"],
        'lr': params['lr_emb'],
    }
    nn_group = {
        'params': nn_params,
        'weight_decay': params["l2_nn"],
        'lr':params['lr_nn'],
    }

    optimizer = torch.optim.Adam([embedding_group, nn_group])

    return optimizer

def save_npy_record(npy_path, record):
    max_index = 0
    for filename in os.listdir(npy_path):
        if filename.startswith("record") and filename.endswith(".npy"):
            max_index +=1

    if max_index==0:
        np.save(npy_path+'/record.npy', record)
    else:
        np.save(npy_path+'/record_{}.npy'.format(max_index), record)


if __name__ == "__main__":
    print("done")