import torch
from torch.nn import functional as F
import numpy as np

class BaseRecommender(torch.nn.Module):
    def __init__(self,config):
        super(BaseRecommender, self).__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = 32 + config['item_embedding_dim']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.dropout = config['dropout']

        self.vars = torch.nn.ParameterDict()
        self.vars_bn = torch.nn.ParameterList()

        self.vars['ml_fc_w1'] = self.get_initialed_para_matrix(self.fc2_in_dim, self.fc1_in_dim)
        self.vars['ml_fc_b1'] = self.get_zero_para_bias(self.fc2_in_dim)

        self.vars['ml_fc_w2'] = self.get_initialed_para_matrix(self.fc2_out_dim, self.fc2_in_dim)
        self.vars['ml_fc_b2'] = self.get_zero_para_bias(self.fc2_out_dim)

        self.vars['ml_fc_w3'] = self.get_initialed_para_matrix(1, self.fc2_out_dim)
        self.vars['ml_fc_b3'] = self.get_zero_para_bias(1)

    def get_zero_para_bias(self, num):
        return torch.nn.Parameter(torch.zeros(num))

    def get_initialed_para_matrix(self, out_num, in_num):
        w = torch.nn.Parameter(torch.ones([out_num, in_num]))
        torch.nn.init.xavier_normal_(w)
        return w

    def forward(self, user_emb, item_emb, user_neigh_emb, vars_dict=None):
        """
        """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_neigh_emb

        x = torch.cat((x_i, x_u), 1)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        # x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        # x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

    def get_parameter_size(self):
        parameter_name_size = dict()
        for key in self.vars.keys():
            weight_size = np.prod(self.vars[key].size())
            parameter_name_size[key] = weight_size

        return parameter_name_size

class MataPathRecommender(torch.nn.Module):
    def __init__(self,config):
        super(MataPathRecommender, self).__init__()
        self.config = config

        self.vars = torch.nn.ParameterDict()
        neigh_w = torch.nn.Parameter(torch.ones([32,config['item_embedding_dim']]))
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, user_emb, item_emb, neighs_emb, mp, index_list, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        output_emb = F.leaky_relu(torch.mean(agg_neighbor_emb, 0)).repeat(user_emb.shape[0], 1)
        return output_emb

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars
