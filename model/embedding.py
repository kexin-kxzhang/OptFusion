import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, feature_num, emb_dim, std=1e-4):
        super(Embedding, self).__init__()
        self.feature_num = feature_num
        self.emb_dim = emb_dim
        self.weight = torch.nn.Parameter(torch.rand(self.feature_num, self.emb_dim))
        torch.nn.init.normal_(self.weight, mean=0, std=std)

    def forward(self, x):
        out = F.embedding(x, self.weight)
        return out
