import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding import Embedding
import util.utils as utils
import os

class ConcatPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConcatPooling, self).__init__()
        self.concat_pooling = nn.Linear(input_dim, output_dim)

    def forward(self, *inputs):
        if len(inputs) not in [2, 4, 6, 8]:
            raise ValueError("Supported number: 2, 4, 6, 8")

        concat_input = torch.cat(inputs, dim=-1)
        return self.concat_pooling(concat_input)
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, layer_index):
        super(AttentionPooling, self).__init__()
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.Softmax(dim=-1)
            )
            for _ in range(layer_index * 2)
        ])

    def forward(self, *inputs):
        if len(inputs) not in [2, 4, 6, 8]:
            raise ValueError("Supported number: 2, 4, 6, 8")

        result = 0
        for i in range(len(inputs)):
            result += self.attentions[i](inputs[i]) * inputs[i]

        return result

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation

class MLP_Layer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 dropout_rates=0.0, 
                 batch_norm=False, 
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        self.dnn = nn.Sequential(*dense_layers) 
    
    def forward(self, inputs):
        return self.dnn(inputs)
    
class CrossNetwork_Layer(torch.nn.Module):  
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias + X_i
        return interaction_out
    
class AutoFusion_train(torch.nn.Module):
    def __init__(self, data_config, model_config):
        super(AutoFusion_train, self).__init__()
        self.load_path = model_config['load']
        self.emb_dim = model_config['emb_dim']
        self.feature_num = data_config['feature_num']
        self.field_num = data_config['field_num']
        self.mlp_dims = model_config['mlp_dims']
        self.dropout = model_config['mlp_dropout']
        self.use_bn = model_config['use_bn']
        self.embed_output_dim = self.field_num * self.emb_dim
        self.embedding = Embedding(data_config['feature_num'], model_config['emb_dim'], model_config['emb_std'])

        self.fusion_mode = model_config['fusion_mode']
        self.device = torch.device('cuda:'+str(0))
        self.num_layers = len(self.mlp_dims) # num_blocks
        self.fusion_types = ["pointwise_addition", "hadamard_product", "concatenation", "attention_pooling"]
        self.hidden_activations="ReLU"

        self.dense_layers = nn.ModuleList([MLP_Layer(self.embed_output_dim, # MLP = linear + activation + dropout
                                               output_dim=None,
                                               hidden_units=[self.embed_output_dim], 
                                               hidden_activations=self.hidden_activations,
                                               dropout_rates=self.dropout, 
                                               batch_norm=self.use_bn) \
                                           for _ in range(len(self.mlp_dims))])
        self.cross_layer_zero = CrossNetwork_Layer(self.embed_output_dim)
        self.cross_layers = nn.ModuleList([CrossNetwork_Layer(self.embed_output_dim) for _ in range(len(self.mlp_dims))])
        self.fc = torch.nn.Linear(self.embed_output_dim, 1)
        
        self.loaded_connection, self.loaded_fusion = self.load_arch_from_search()
        self.initialize_fusion_params()

    def forward(self, x):
        xv = self.embedding(x)
        emb = xv.view(-1, self.embed_output_dim)
        cross_0 = emb
        cross_outputs=[]
        deep_outputs=[]
        
        # cross 0 layer
        cross_zero_input = emb * self.loaded_connection[1, 0]
        cross_zero_output = self.cross_layer_zero(cross_0, cross_zero_input)

        for i in range(self.num_layers):
            cross_i_inputs=[]
            deep_i_inputs=[]
            # 2 & 3 blocks
            if i > 0:
                cross_i_inputs.append(emb * self.loaded_connection[2*(i+1), 0])
                cross_i_inputs.append(cross_zero_output * self.loaded_connection[2*(i+1), 1])
                for j in range(i):
                    cross_i_inputs.append(cross_outputs[j] * self.loaded_connection[2*(i+1),(j+1)*2])
                    cross_i_inputs.append(deep_outputs[j] * self.loaded_connection[2*(i+1),(j+1)*2+1])

                cross_i_input_fused = self.compute_fusion_result(self.loaded_fusion[2*i], cross_i_inputs, 2*i)
                cross_out = self.cross_layers[i](cross_0, cross_i_input_fused)
                    
                deep_i_inputs.append(emb * self.loaded_connection[2*(i+1)+1, 0])
                deep_i_inputs.append(cross_zero_output * self.loaded_connection[2*(i+1)+1, 1])
                for j in range(i):
                    deep_i_inputs.append(cross_outputs[j] * self.loaded_connection[2*(i+1)+1, (j+1)*2])
                    deep_i_inputs.append(deep_outputs[j] * self.loaded_connection[2*(i+1)+1, (j+1)*2+1])

                deep_i_input_fused = self.compute_fusion_result(self.loaded_fusion[2*i+1], deep_i_inputs, 2*i+1)            
                deep_out = self.dense_layers[i](deep_i_input_fused)
            
            # 1 block
            else:
                cross_i_inputs.append(emb * self.loaded_connection[2*(i+1), 0]) 
                cross_i_inputs.append(cross_zero_output * self.loaded_connection[2*(i+1), 1])
                cross_i_input_fused = self.compute_fusion_result(self.loaded_fusion[2*i], cross_i_inputs, 2*i)
                cross_out = self.cross_layers[i](cross_0, cross_i_input_fused)

                deep_i_inputs.append(emb * self.loaded_connection[2*(i+1)+1, 0])
                deep_i_inputs.append(cross_zero_output * self.loaded_connection[2*(i+1)+1, 1])
                deep_i_input_fused = self.compute_fusion_result(self.loaded_fusion[2*i+1], deep_i_inputs, 2*i+1)
                deep_out = self.dense_layers[i](deep_i_input_fused)

            cross_outputs.append(cross_out)
            deep_outputs.append(deep_out)
        
        # Output block
        final_inputs = []
        final_inputs.append(emb * self.loaded_connection[-1, 0])
        final_inputs.append(cross_zero_output * self.loaded_connection[-1, 1])
        for j in range(self.num_layers):
            final_inputs.append(cross_outputs[j] * self.loaded_connection[-1, (j+1) * 2])
            final_inputs.append(deep_outputs[j] * self.loaded_connection[-1, (j+1) * 2 + 1])
        final_input_fused = self.compute_fusion_result(self.loaded_fusion[-1], final_inputs, 6)
        
        score = self.fc(final_input_fused)
        score = score.squeeze(1)
        
        return score

    
    def initialize_fusion_params(self):
        self.concat_pooling_c1 = ConcatPooling(self.embed_output_dim * 2, self.embed_output_dim)
        self.concat_pooling_d1 = ConcatPooling(self.embed_output_dim * 2, self.embed_output_dim)
        self.concat_pooling_c2 = ConcatPooling(self.embed_output_dim * 4, self.embed_output_dim)
        self.concat_pooling_d2 = ConcatPooling(self.embed_output_dim * 4, self.embed_output_dim)
        self.concat_pooling_c3 = ConcatPooling(self.embed_output_dim * 6, self.embed_output_dim)
        self.concat_pooling_d3 = ConcatPooling(self.embed_output_dim * 6, self.embed_output_dim)
        self.concat_pooling_last = ConcatPooling(self.embed_output_dim * 8, self.embed_output_dim)

        self.attention_pooling_c1 = AttentionPooling(self.embed_output_dim, 1)
        self.attention_pooling_d1 = AttentionPooling(self.embed_output_dim, 1)
        self.attention_pooling_c2 = AttentionPooling(self.embed_output_dim, 2)
        self.attention_pooling_d2 = AttentionPooling(self.embed_output_dim, 2)
        self.attention_pooling_c3 = AttentionPooling(self.embed_output_dim, 3)
        self.attention_pooling_d3 = AttentionPooling(self.embed_output_dim, 3)
        self.attention_pooling_last = AttentionPooling(self.embed_output_dim, 4)


    def compute_fusion_result(self, fusion_params_init, inputs, block_index):
        if self.fusion_mode == 1:
            fusion_params = F.softmax(fusion_params_init, dim=1).to(self.device)
            result_add = torch.sum(torch.stack(inputs), dim=0)
            result_prod = torch.prod(torch.stack(inputs), dim=0)
            if block_index == 0:
                result_concat = self.concat_pooling_c0(*inputs)
                result_atten = self.attention_pooling_c0(*inputs)
            if block_index == 1:
                result_concat = self.concat_pooling_d0(*inputs)
                result_atten = self.attention_pooling_d0(*inputs)
            if block_index == 2:
                result_concat = self.concat_pooling_c1(*inputs)
                result_atten = self.attention_pooling_c1(*inputs)
            if block_index == 3:
                result_concat = self.concat_pooling_d1(*inputs)
                result_atten = self.attention_pooling_d1(*inputs)
            if block_index == 4:
                result_concat = self.concat_pooling_c2(*inputs)
                result_atten = self.attention_pooling_c2(*inputs)
            if block_index == 5:
                result_concat = self.concat_pooling_d2(*inputs)
                result_atten = self.attention_pooling_d2(*inputs)
            if block_index == 6:
                result_concat = self.concat_pooling_last(*inputs)
                result_atten = self.attention_pooling_last(*inputs)

            result = (
                fusion_params[:, 0].unsqueeze(1).mul(result_add) +
                fusion_params[:, 1].unsqueeze(1).mul(result_prod) +
                fusion_params[:, 2].unsqueeze(1).mul(result_concat) +
                fusion_params[:, 3].unsqueeze(1).mul(result_atten)
            )

            return result
            
        elif self.fusion_mode == 2:
            fusion_params = F.softmax(fusion_params_init, dim=1)
            fusion_index = torch.argmax(fusion_params).item() 
            fusion_type = self.fusion_types[fusion_index]

            if fusion_type == "pointwise_addition":
                result = torch.sum(torch.stack(inputs), dim=0)
            elif fusion_type == "hadamard_product":
                non_zero_inputs = [input for input in inputs if torch.sum(torch.abs(input)) != 0]
                print(len(non_zero_inputs))
                if len(non_zero_inputs) > 0:
                    result = torch.prod(torch.stack(non_zero_inputs), dim=0)
                else:
                    result = torch.zeros_like(inputs[0])
            elif fusion_type == "concatenation":
                if block_index == 0:
                    result = self.concat_pooling_c1(*inputs)
                if block_index == 1:
                    result = self.concat_pooling_d1(*inputs)
                if block_index == 2:
                    result = self.concat_pooling_c2(*inputs)
                if block_index == 3:
                    result = self.concat_pooling_d2(*inputs)
                if block_index == 4:
                    result = self.concat_pooling_c3(*inputs)
                if block_index == 5:
                    result = self.concat_pooling_d3(*inputs)
                if block_index == 6:
                    result = self.concat_pooling_last(*inputs)
            elif fusion_type == "attention_pooling":
                if block_index == 0:
                    result = self.attention_pooling_c1(*inputs)
                if block_index == 1:
                    result = self.attention_pooling_d1(*inputs)
                if block_index == 2:
                    result = self.attention_pooling_c2(*inputs)
                if block_index == 3:
                    result = self.attention_pooling_d2(*inputs)
                if block_index == 4:
                    result = self.attention_pooling_c3(*inputs)
                if block_index == 5:
                    result = self.attention_pooling_d3(*inputs)
                if block_index == 6:
                    result = self.attention_pooling_last(*inputs)

            return result
        
        else:
            return torch.sum(torch.stack(inputs), dim=0)
    
    def load_arch_from_search(self):
        self.searched_connection_arch = utils.load_arch(os.path.join(self.load_path, 'arch_weight.npy'))
        
        self.loaded_connection_from_search = torch.from_numpy(self.searched_connection_arch)
        self.loaded_connection = torch.where(self.loaded_connection_from_search > 0, torch.tensor(1), torch.tensor(0)) 
        print("self.loaded_connection:", self.loaded_connection) #9*9

        self.searched_fusion_0_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_0.npy'))
        self.loaded_fusion_0 = torch.from_numpy(self.searched_fusion_0_arch)

        self.searched_fusion_1_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_1.npy'))
        self.loaded_fusion_1 = torch.from_numpy(self.searched_fusion_1_arch)

        self.searched_fusion_2_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_2.npy'))
        self.loaded_fusion_2 = torch.from_numpy(self.searched_fusion_2_arch)

        self.searched_fusion_3_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_3.npy'))
        self.loaded_fusion_3 = torch.from_numpy(self.searched_fusion_3_arch)

        self.searched_fusion_4_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_4.npy'))
        self.loaded_fusion_4 = torch.from_numpy(self.searched_fusion_4_arch)

        self.searched_fusion_5_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_5.npy'))
        self.loaded_fusion_5 = torch.from_numpy(self.searched_fusion_5_arch)

        self.searched_fusion_6_arch = utils.load_arch(os.path.join(self.load_path, 'fusion_weight_6.npy'))
        self.loaded_fusion_6 = torch.from_numpy(self.searched_fusion_6_arch)

        self.loaded_fusion = [self.loaded_fusion_0, self.loaded_fusion_1, self.loaded_fusion_2, self.loaded_fusion_3, self.loaded_fusion_4, 
                              self.loaded_fusion_5, self.loaded_fusion_6]

        return self.loaded_connection, self.loaded_fusion
