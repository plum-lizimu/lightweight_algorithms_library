import torch
import torch.nn as nn
import math

class Base(nn.Module):

    '''
        init.
        features:
            feature_name, type_number  e.g. user & user_num
        device:
            cpu / cuda
        given_dims:
            feature_name, embedding_dims
    '''
    def __init__(self, params):
        super().__init__()
        self.base ='base'
        self.features_count, self.given_dims, self.device = params['features'], params['given_dims'], params['device']

        emb_dict = {}
        self.dim = 0
        self.feat_dim = 0
        for k, n in self.features_count.items():
            if self.given_dims is not None:
                dim = self.given_dims[k]
            else:
                dim = int(math.log(n) + 8)
            emb_dict[k] = nn.Embedding(n, dim)
            self.dim = max(self.dim, dim)
            self.feat_dim += dim
        self.embeddings = nn.ModuleDict(emb_dict)

    '''
       Retrieve embedding representations of the current data.
       discrete=True : retrieving the embedding representation of discrete data
       discrete=False: retrieving the embedding representation of continuous data

       batch_feature:
        discrete data(one-hot encoding):
            user: [0, 3, 1, 2] 
                  <=>
                  [[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]]

            item: [1, 0, 3, 2]
                  <=>
                  [[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]

        continuous data
            user: [[0.1, 0.2, 0.3, 0.4],
                   [0.2, 0.1, 0.4, 0.3],
                   [0.5, 0.1, 0.1, 0.3],
                   [0.6, 0.1, 0.2, 0.1]]
            
            item: [[0.2, 0.2, 0.2, 0.3],
                   [0.25, 0.25, 0.25, 0.25],
                   [0.1, 0.1, 0.1, 0.7],
                   [0.2, 0.1, 0.3, 0.4]]
    '''
    def get_embedding(self, batch_feature, discrete):
        feature_embedding_list = []
        for k, emb in self.embeddings.items():
            if discrete:
                feature_emb = emb(batch_feature[k])
            else:
                emb_weight = batch_feature[k]
                batch_size = emb_weight.size(0)
                all = torch.tensor([e for e in range(self.features_count[k])]).to(self.device)
                batch_all = all.repeat(batch_size).view(-1, self.features_count[k])
                feature_emb = emb(batch_all)
                emb_weight = emb_weight.unsqueeze(-1)
                feature_emb.mul_(emb_weight)
                # feature_emb = (feature_emb * emb_weight).sum(1)
                feature_emb = feature_emb.sum(1)
            feature_embedding_list.append(feature_emb)
        return feature_embedding_list
    