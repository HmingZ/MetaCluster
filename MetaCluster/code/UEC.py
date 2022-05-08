import torch.nn as F
import torch.nn.init as init
from utils import *

class TaskEncoder(F.Module):
    def __init__(self, config):
        super(TaskEncoder, self).__init__()
        self.x_dim = config['x_dim']
        self.y_dim = 0 # or 1
        self.h1_dim = config['taskenc_h1_dim']
        self.h2_dim = config['taskenc_h2_dim']
        self.final_dim = config['taskenc_final_dim']
        self.dropout_rate = config['dropout_rate']
        self.config = config
        layers = [F.Linear(self.x_dim + self.y_dim, self.h1_dim),
                  F.Dropout(self.dropout_rate),
                  F.ReLU(inplace=True),
                  F.Linear(self.h1_dim, self.h2_dim),
                  F.Dropout(self.dropout_rate),
                  F.ReLU(inplace=True),
                  F.Linear(self.h2_dim, self.final_dim)]
        self.task_mlp = F.Sequential(*layers)

    def forward(self, x, y):
        # y = y.view(-1, 1)
        # catxy = torch.cat((x, y), dim=1)
        return self.task_mlp(x)  #or catxy

class TaskClustering(F.Module):
    def __init__(self, config):
        super(TaskClustering, self).__init__()
        self.clusters_k = config['clusters_k']
        self.embed_size = config['taskenc_final_dim']
        self.alpha = config['alpha']
        self.cluster_array = F.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, task_embedding):
        res = torch.norm(task_embedding - self.cluster_array, p=2, dim=1, keepdim=True)
        assignments = torch.pow((res / self.alpha) + 1, (self.alpha + 1) / -2)
        A = torch.transpose(assignments / assignments.sum(), 0, 1)
        Fu = torch.mm(A, self.cluster_array)
        new_task_embeding = Fu + task_embedding
        return A, new_task_embeding
        
class TaskMem:
    def __init__(self, n_k, emb_dim):
        self.n_k = n_k
        self.memory_UI = torch.rand(n_k, emb_dim *2, emb_dim*2).normal_()
        self.att_values = torch.zeros(n_k)

    def read_head(self, att_values):
        self.att_values = att_values
        return get_mui(att_values, self.memory_UI, self.n_k)

    def write_head(self, u_mui, lr):
        update_values = update_mui(self.att_values, self.n_k, u_mui)
        self.memory_UI = lr* update_values + (1-lr) * self.memory_UI



def get_mui(att_values, mui, n_k):
    attention_values = att_values.reshape(n_k, 1, 1)
    attend_mui = torch.mul(attention_values, mui)
    u_mui = attend_mui.sum(dim=0)
    return u_mui


def update_mui(att_values, n_k, u_mui):
    repeat_u_mui = u_mui.unsqueeze(0).repeat(n_k, 1, 1)
    attention_tensor = att_values.reshape(n_k, 1, 1)
    attend_u_mui = torch.mul(attention_tensor, repeat_u_mui)
    return attend_u_mui