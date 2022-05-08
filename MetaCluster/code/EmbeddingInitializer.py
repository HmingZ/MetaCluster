import torch
from torch.autograd import Variable

class UserEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingML, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']

        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        gender_idx = Variable(user_fea[:, 0], requires_grad=False)
        age_idx = Variable(user_fea[:, 1], requires_grad=False)
        occupation_idx = Variable(user_fea[:, 2], requires_grad=False)
        area_idx = Variable(user_fea[:, 3], requires_grad=False)

        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)


class ItemEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, item_fea):
        rate_idx = Variable(item_fea[:, 0], requires_grad=False)
        genre_idx = Variable(item_fea[:, 1:26], requires_grad=False)
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)  # (1,32)
        return torch.cat((rate_emb, genre_emb), 1)
