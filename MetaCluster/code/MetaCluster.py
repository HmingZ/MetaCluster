import numpy as np
import torch
from torch.nn import functional as F
from Evaluation import Evaluation
from Recommender import MataPathRecommender, BaseRecommender
from Task_Modulation import Task_modulation
from UEC import TaskEncoder, TaskClustering
from utils import *

class MetaCluster(torch.nn.Module):
    def __init__(self, config, model_name):
        super(MetaCluster, self).__init__()
        self.model_name = model_name
        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self._lambda = config['lambda']
        self._alpha = config['alpha']

        if self.config['dataset'] == 'movielens':
            from EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML
            self.item_emb = ItemEmbeddingML(config)
            self.user_emb = UserEmbeddingML(config)

        self.mp_recommender = MataPathRecommender(config)
        self.base_recommender = BaseRecommender(config)
        self.task_fc = TaskEncoder(config)
        self.taskclustering = TaskClustering(config)
        self.local_lr = config['local_lr']
        self.emb_dim = self.config['embedding_dim']
        self.input_dim = self.config['input_dim']
        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.base_recommender.update_parameters())
        self.ml_weight_name = list(self.base_recommender.update_parameters().keys())
        self.mp_weight_len = len(self.mp_recommender.update_parameters())
        self.mp_weight_name = list(self.mp_recommender.update_parameters().keys())
        self.ml_weight_size = self.base_recommender.get_parameter_size()
        self.task_modulation = Task_modulation(config, self.ml_weight_size)
        self.transformer_liners = self.transform_mp2task()
        # global update
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.base_recommender.update_parameters()
        output_dim_of_mp = 32
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp,
                                                          np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def task_aggregate(self, z_i):
        return torch.mean(z_i, dim=0)

    def forward(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.base_recommender.update_parameters()

        support_set_y_pred = self.base_recommender(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)

        fast_weights = {}
        for idx, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[idx]

        for idx in range(1, self.config['local_update']):
            support_set_y_pred = self.base_recommender(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(loss, fast_weights.values(),
                                       create_graph=True)

            for idx, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[idx]

        return fast_weights

    def metaCluster(self, support_set_x, support_set_y, support_set_mps, query_set_x, query_set_y, query_set_mps):
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        support_user_emb = self.user_emb(support_set_x[:, self.config['item_fea_len']:])
        support_item_emb = self.item_emb(support_set_x[:, 0:self.config['item_fea_len']])
        query_user_emb = self.user_emb(query_set_x[:, self.config['item_fea_len']:])
        query_item_emb = self.item_emb(query_set_x[:, 0:self.config['item_fea_len']])

        for mp in self.config['mp']:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = map(lambda _: _.shape[0], support_set_mp)
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)

            support_mp_enhanced_user_emb = self.mp_recommender(support_user_emb, support_item_emb, support_neighs_emb, mp,
                                                               support_index_list)
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_recommender(query_user_emb, query_item_emb, query_neighs_emb, mp,
                                                             query_index_list)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)
        agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        support_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        x_embed = torch.cat((support_agg_enhanced_user_emb, support_user_emb), 1)
        task_emb = self.task_fc(x_embed, support_set_y)
        mean_task = self.task_aggregate(task_emb)
        A_distribution, new_task_embedding = self.taskclustering(mean_task)
        ml_initial_weights = self.base_recommender.update_parameters()
        task_weights = self.task_modulation(task_emb, ml_initial_weights)
        # f_fast_weights = {}
        # for w, liner in self.transformer_liners.items():
        #     w = w.replace('-', '.')
        #     f_fast_weights[w] = ml_initial_weights[w] * \
        #                         torch.sigmoid(liner(new_task_embedding)). \
        #                             view(ml_initial_weights[w].shape)

        task_fast_weights = self.forward(support_user_emb, support_item_emb, support_set_y,
                                         support_agg_enhanced_user_emb, vars_dict=task_weights)
        query_y_pred = self.base_recommender(query_user_emb, query_item_emb, query_agg_enhanced_user_emb, vars_dict=task_fast_weights)
        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)

        return loss, mae, rmse, ndcg_5, A_distribution

    def global_update(self, support_xs, support_ys, support_mps, query_xs, query_ys, query_mps, device='cpu'):
        batch_sz = len(support_xs)
        loss_s, mae_s, rmse_s, ndcg_at_5_s, UEC_A_Distribution = [], [], [], [], []

        for i in range(batch_sz):
            support_mp = dict(support_mps[i])
            query_mp = dict(query_mps[i])

            for mp in self.config['mp']:
                support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
                query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])
            _loss, _mae, _rmse, _ndcg_5, _A_distribution = self.metaCluster(support_xs[i].to(device), support_ys[i].to(device), support_mp,
                                                                            query_xs[i].to(device), query_ys[i].to(device), query_mp)
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)
            UEC_A_Distribution.append(_A_distribution)

        UEC_A_Distribution = torch.stack(UEC_A_Distribution)
        cluster_weight_distribution = torch.pow(UEC_A_Distribution, 2) / torch.sum(UEC_A_Distribution, dim=0, keepdim=True)
        cluster_weight_distribution_sum = torch.sum(cluster_weight_distribution, dim=1, keepdim=True)
        UEC_B_Distribution = cluster_weight_distribution / cluster_weight_distribution_sum
        Clustering_Loss = self._lambda * F.kl_div(UEC_A_Distribution.log(), UEC_B_Distribution, reduction='batchmean')

        Prediction_Loss = torch.stack(loss_s).mean(0)
        Total_Loss = Prediction_Loss + Clustering_Loss

        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)
        self.meta_optimizer.zero_grad()
        Total_Loss.backward()
        self.meta_optimizer.step()
        return Total_Loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(self, support_x, support_y, support_mp, query_x, query_y, query_mp, device='cpu'):
        support_mp = dict(support_mp)
        query_mp = dict(query_mp)
        for mp in self.config['mp']:
            support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
            query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])
        _, mae, rmse, ndcg_5, _ = self.metaCluster(support_x.to(device), support_y.to(device), support_mp,
                                                   query_x.to(device), query_y.to(device), query_mp)
        return mae, rmse, ndcg_5
