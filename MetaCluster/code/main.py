import random
import time
import numpy as np
import torch
from MetaCluster import MetaCluster
from DataHelper import DataHelper
from Config import states
random.seed(33)
np.random.seed(33)
torch.manual_seed(33)


def training(model, model_save=True, model_file=None, device='cpu'):
    print('training model...')
    if config['use_cuda']:
        model.cuda()
    model.train()

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']

    for _ in range(num_epoch):
        loss, mae, rmse = [], [], []
        ndcg_at_5 = []
        start = time.time()

        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size)
        supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*train_data)
        for i in range(num_batch):
            support_xs = list(supp_xs_s[batch_size * i:batch_size * (i + 1)])
            support_ys = list(supp_ys_s[batch_size * i:batch_size * (i + 1)])
            support_mps = list(supp_mps_s[batch_size * i:batch_size * (i + 1)])
            query_xs = list(query_xs_s[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i:batch_size * (i + 1)])
            query_mps = list(query_mps_s[batch_size * i:batch_size * (i + 1)])

            _loss, _mae, _rmse, _ndcg_5 = model.global_update(support_xs,support_ys,support_mps,
                                                              query_xs,query_ys,query_mps,device)
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_5.append(_ndcg_5)

        print('epoch: {}, loss: {:.6f}, cost time: {:.1f}s, mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
              format(_, np.mean(loss), time.time() - start,
                     np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)))
        if _ % 10 == 0 and _ != 0:
            testing(model, device)
            model.train()

    if model_save:
        print('saving model...')
        torch.save(model.state_dict(), model_file)


def testing(model, device='cpu'):
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()
    for state in states:
        if state == 'meta_training':
            continue
        print(state + '...')
        evaluate(model, state, device)


def evaluate(model, state, device='cpu'):
    test_data = data_helper.load_data(data_set=data_set, state=state,
                                      load_from_file=True)
    supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(*test_data)
    loss, mae, rmse = [], [], []
    ndcg_at_5 = []

    for i in range(len(test_data)):
        _mae, _rmse, _ndcg_5 = model.evaluation(supp_xs_s[i], supp_ys_s[i], supp_mps_s[i],
                                                query_xs_s[i], query_ys_s[i], query_mps_s[i],device)
        mae.append(_mae)
        rmse.append(_rmse)
        ndcg_at_5.append(_ndcg_5)
    print('mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}'.
          format(np.mean(mae), np.mean(rmse),np.mean(ndcg_at_5)))


if __name__ == "__main__":
    data_set = 'movielens'
    input_dir = '../data/'
    output_dir = '../data/'
    res_dir = '../res/'+data_set
    load_model = False

    from Config import config_ml as config
    cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(config)

    model_filename = "{}/mc.pkl".format(res_dir)
    data_helper = DataHelper(input_dir, output_dir, config)

    model_name = 'metaCluster'
    mc = MetaCluster(config, model_name)

    print('--------------- {} ---------------'.format(model_name))

    if not load_model:
        print('loading train data...')
        train_data = data_helper.load_data(data_set=data_set,state='meta_training',load_from_file=True)
        training(mc, model_save=True, model_file=model_filename, device=cuda_or_cpu)
    else:
        trained_state_dict = torch.load(model_filename)
        mc.load_state_dict(trained_state_dict)

    testing(mc, device=cuda_or_cpu)
    print('--------------- {} ---------------'.format(model_name))