import argparse
from util.tool import get_data_loader
import util.utils as utils
from train.search import Search
import tensorflow as tf
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
import numpy as np
import glob
import logging
import time
import sys
current_date = time.strftime("%Y%m%d-%H%M%S")
tf.compat.v1.enable_eager_execution()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
# args for model.
parser.add_argument('--dataset', type=str, default='avazu_2', help='dataset name: criteo/avazu')
parser.add_argument('--model', type=str, default='optfusion_search', help='model: optfusion_search/optfusion_train')
parser.add_argument('--emb_dim', type=int, default=40, help='embedding dimension')
parser.add_argument('--emb_std', type=float, default=1e-2, help='dropout for mlp')
parser.add_argument('--mlp_dims', type=int, nargs='+', default=[960, 960, 960], help='dimension for each layer')
parser.add_argument('--mlp_dropout', type=float, default=0.0, help='dropout for mlp')
parser.add_argument('--use_bn', action='store_true', default=False, help='bn')
parser.add_argument('--fusion_type', type=str, default='concatenation', help='fusion type of cross and deep block')
parser.add_argument('--cross_method', type=str, default='Mix', help='DCN_v2: Mix or Metrix')
parser.add_argument('--model_method', type=str, default='parallel', help='DCN_v2: parallel or stack')
# args for training.
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
parser.add_argument('--batch_size', type=float, default=4096, help='batch size')
parser.add_argument('--lr_emb', type=float, default=3e-3, help='learning rate')
parser.add_argument('--lr_nn', type=float, default=3e-3, help='learning rate')
parser.add_argument('--arch_lr', type=float, default=1e-1, 
                    help='learning rate for arch encoding') # new add arch lr
parser.add_argument('--l2_emb', type=float, default=3e-6, help='weight decay for embedding table')
parser.add_argument('--l2_nn', type=float, default=3e-6, help='weight decay for embedding table')
parser.add_argument('--init_constant', type=float, default=0.5, help='init connection params')
parser.add_argument('--epochs', type=int, default=5, help='max epoch for training')
parser.add_argument('--save_step', action='store_true', default=False, help='save step checkpoint')
parser.add_argument('--val_per_epoch', type=int, default=5, help='')
parser.add_argument('--early_stop', type=int, default=1, help='how many epochs to stop')
parser.add_argument('--tau', type=float, default=1.0, help='edcn for regulation')
parser.add_argument('--fusion_mode', type=int, default=1,
                    help='whether 4 fusion types or only add, 1 indicates 4 fusion types, others indicate only add')
# args for log.
parser.add_argument('--log_path', type=str, default='./log/', help='log file save path.')

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=0, help="device info")
args = parser.parse_args()
log_name = f'{args.dataset}_{args.model}_{current_date}_{"_".join(map(str, args.mlp_dims))}_fusion_mode{args.fusion_mode}_init_constant{args.init_constant}_lr_emb{args.lr_emb}_lr_nn{args.lr_nn}_arch_lr{args.arch_lr}_l2_emb{args.l2_emb}_l2_nn{args.l2_nn}_batch_size{args.batch_size}'
args.log_name = log_name

my_seed = 2024
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)


data_loader, data_config = get_data_loader(args.dataset)
model_config = {
    "model": args.model,
    "emb_dim": args.emb_dim,
    "emb_std": args.emb_std,
    "mlp_dims": args.mlp_dims,
    "mlp_dropout": args.mlp_dropout,
    "use_bn": args.use_bn,
    "fusion_type": args.fusion_type,
    "cross_method": args.cross_method,
    "model_method": args.model_method,
    "tau": args.tau,
    "fusion_mode": args.fusion_mode,
    "init_constant": args.init_constant,
}
train_config = {
    "optimizer": args.optimizer,
    "batch_size": int(args.batch_size),
    "lr_emb": args.lr_emb,
    "lr_nn": args.lr_nn,
    "l2_emb": args.l2_emb,
    "l2_nn": args.l2_nn,
    "arch_lr": args.arch_lr,
    "epochs": args.epochs,
    "save_step": args.save_step,
    "val_per_epoch": args.val_per_epoch,
    "early_stop": args.early_stop,
    "cuda": args.cuda,
    "log_path": args.log_path,
    "log_name": args.log_name,
}
search = Search(data_loader, data_config, model_config, train_config)
search.train_epochs()




