import gc
import glob
import os
import pickle
from tqdm import tqdm

class DataHelper:
    def __init__(self, input_dir, output_dir, config):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.mp_list = self.config['mp']

    def load_data(self, data_set, state, load_from_file=True):
        data_dir = os.path.join(self.output_dir,data_set)
        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        training_set_size = int(len(glob.glob("{}/{}/*.pkl".format(data_dir,state))) / self.config['file_num'])

        for idx in tqdm(range(training_set_size)):
            support_x = pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb"))
            if support_x.shape[0] > 3:
                continue
            del support_x
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            supp_mp_data, query_mp_data = {}, {}
            for mp in self.mp_list:
                supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
            supp_mps_s.append(supp_mp_data)
            query_mps_s.append(query_mp_data)

        print('#support set: {}, #query set: {}'.format(len(supp_xs_s), len(query_xs_s)))
        total_data = list(zip(supp_xs_s, supp_ys_s, supp_mps_s,
                              query_xs_s, query_ys_s, query_mps_s))  # all training tasks
        del (supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s)
        gc.collect()
        return total_data

    def get_batch_data(self, data_set, state, batch_indices, load_from_file=True):
        data_dir = os.path.join(self.output_dir,data_set)

        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        for idx in batch_indices:
            supp_xs_s.append(pickle.load(open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")))

            supp_mp_data, query_mp_data = {}, {}
            for mp in self.mp_list:
                supp_mp_data[mp] = pickle.load(open("{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))
                query_mp_data[mp] = pickle.load(open("{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx), "rb"))

            supp_mps_s.append(supp_mp_data)
            query_mps_s.append(query_mp_data)
        return {'supp_xs_s': supp_xs_s,
                'supp_ys_s': supp_ys_s,
                'query_xs_s': query_xs_s,
                'query_ys_s': query_ys_s,
                'supp_mps_s': supp_mps_s,
                'query_mps_s': query_mps_s
                }