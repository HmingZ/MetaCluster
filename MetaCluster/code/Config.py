config_ml = {
    'dataset': 'movielens',
    'mp': ['um','umum','umam','umdm'],
    'use_cuda': True,
    'gpu': '2',
    'file_num': 12,

    'num_rate': 6,
    'num_genre': 25,
    'num_fea_item': 2,
    'item_fea_len': 26,

    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 4,

    'embedding_dim': 32,
    'user_embedding_dim': 32*4,
    'item_embedding_dim': 32*2,

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    'local_update': 1,  #1-5
    'lr': 5e-4,
    'local_lr': 5e-3,   #1e-3
    'batch_size': 32, #or 64
    'num_epoch': 120,   #or others

    'dropout_rate': 0.2,
    'taskenc_h1_dim': 128,
    'taskenc_h2_dim': 64,
    'taskenc_final_dim': 32,
    'clusters_k': 8,
    'alpha': 1.0,
    'lambda': 0.1,
    'x_dim': 160,
}

states = ["meta_training","warm_up", "user_cold_testing"]