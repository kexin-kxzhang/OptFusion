import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from util.logger import get_log
from model.models import get_model
from util.tool import get_device, get_optimizer, save_npy_record
import util.utils as utils
import logging
import time

class Search(object):
    def __init__(self, data_loader, data_config, model_config, train_config):
        self.data_loader, self.data_config = data_loader, data_config
        self.model_config, self.train_config = model_config, train_config
        # model
        self.device = get_device(train_config['cuda'])
        self.model = get_model(model_config, data_config)
        
        self.model.to(self.device)
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits
        self.optimizer = get_optimizer(self.model, train_config) 
        
        connection_arch_optim_config = {
            'params': self.model.connection_params_init,
            'lr': self.train_config['arch_lr'],
        }
        fusion_arch_optim_config = {
            'params': self.model.fusion_params_init_list,
            'lr': self.train_config['arch_lr'],
        }
        self.arch_optimizer = torch.optim.Adam([connection_arch_optim_config, fusion_arch_optim_config])
        # parameter
        self.sample_nums, self.batch_size = data_config["sample_nums"], train_config["batch_size"]
        self.val_per_epoch, self.early_stop = self.train_config['val_per_epoch'], self.train_config['early_stop']
        # log
        self.log_path = self.train_config["log_path"]
        self.log_name = self.train_config["log_name"]
        self.save_path = os.path.join(self.log_path, self.log_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = get_log(self.log_path, self.log_name)
        self.logger.info("data_config: {}".format(self.data_config))
        self.logger.info("model_config: {}".format(self.model_config))
        self.logger.info("train_config: {}".format(self.train_config))

    def update(self, feature, label, model_config):
        self.model_config = model_config
        self.model.train()
        self.arch_optimizer.zero_grad()
        self.optimizer.zero_grad()
        feature, label = feature.to(self.device), label.to(self.device)
        prob = self.model.forward(feature)
        loss = self.criterion(prob, label.squeeze())  
        loss.backward()
        self.arch_optimizer.step()
        self.optimizer.step()
        return loss.item() 

    def evaluate(self, eval_type="valid", eval_steps=None):
        self.model.eval()
        if eval_steps is None:
            eval_steps = int(self.sample_nums[eval_type] // self.batch_size) + 1
        data_loader = self.data_loader.get_data(eval_type, batch_size=self.batch_size)
        val_prob, val_true = [], []
        for feature, label in tqdm(data_loader, bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}', 
                                   total=eval_steps, ncols=80, desc="{}.iter".format(eval_type)):
            feature = feature.to(self.device)
            prob = self.model.forward(feature)
            prob = torch.sigmoid(prob).detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            val_prob.append(prob)
            val_true.append(label)
        y_prob = np.concatenate(val_prob).astype("float64")
        y_true = np.concatenate(val_true).astype("float64")
        val_auc = round(roc_auc_score(y_true, y_prob), 6)
        val_loss = round(log_loss(y_true, y_prob), 6)
        return val_auc, val_loss

    def train_epochs(self):
        train_steps = self.sample_nums["train"] // self.batch_size + 1
        best_auc, best_loss, best_epoch, best_step = 0.0, 0.0, 0, 0
        starttime = time.time()
        self.logger.info('begin training ...')
        
        for epoch in range(1, self.train_config["epochs"]+1, 1):
            
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}', 
                                       total=train_steps, ncols=80, desc="training.iter"):
                step += 1
                loss = self.update(feature, label, model_config=self.model_config)
            val_auc, val_loss = self.evaluate(eval_type="valid")
            self.logger.info("* valid epoch:{}; step:{}; auc:{:.4f}; loss:{:.4f}.".format(epoch, step, val_auc, val_loss))
            
            if val_auc > best_auc:
                best_auc, best_loss, best_epoch, best_step = val_auc, val_loss, epoch, step

                arch_params = self.model.get_arch_parameters()
                np.save(os.path.join(self.save_path, 'arch_weight.npy'), arch_params["connection_params"].numpy())
                for i, fusion_params in enumerate(arch_params["fusion_params_list"]):
                    np.save(os.path.join(self.save_path, f'fusion_weight_{i}.npy'), fusion_params.numpy())
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))

            if epoch - best_epoch >= self.early_stop or epoch == self.train_config["epochs"]:
                best_dict = torch.load(os.path.join(self.save_path, 'model_weights.pth'))
                self.model.load_state_dict(best_dict)

                test_auc, test_loss = self.evaluate(eval_type="test")
                endtime = time.time()
                self.logger.info("* best epoch:{}; best step:{}; best valid loss: {:.4f}; best valid auc: {:.4f}".format(
                    best_epoch, best_step, best_loss, best_auc))
                self.logger.info("** valid loss:{:.4f}; valid auc: {:.4f}.".format(best_loss, best_auc))
                self.logger.info("** test  loss:{:.4f}; test  auc: {:.4f}.".format(test_loss, test_auc))
                self.logger.info('Time: %f', endtime-starttime)
                record={"loss":[best_loss, test_loss], "auc":[best_auc, test_auc]}
                save_npy_record(self.save_path, record)
                break

