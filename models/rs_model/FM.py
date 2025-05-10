from .Base import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FM(Base):

    def __init__(self, params):
        super().__init__(params)
        self.k = params['k']
        self.LogisticsRegression = nn.Linear(self.feat_dim, 1, bias=True)
        self.oneDeg = nn.Linear(self.feat_dim, 1)
        self.twoDeg_v = nn.parameter.Parameter(torch.zeros(size=(self.feat_dim, self.k)))
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.normal_(weight, 0, 1/math.sqrt(self.dim))
    
    def fm_forward(self, batch_feature, discrete, o_sigmoid):
        feature_embedding_list = self.get_embedding(batch_feature, discrete)
        all_feature = torch.concat(feature_embedding_list, -1)
        result = self.oneDeg(all_feature).squeeze()
        out_1 = torch.matmul(all_feature, self.twoDeg_v).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(all_feature.pow(2), self.twoDeg_v.pow(2)).sum(1, keepdim=True)
        result += (0.5 * (out_1 - out_2)).squeeze()
        if o_sigmoid:
            return F.sigmoid(result)
        return result

    def forward(self, batch_feature, discrete=True, o_sigmoid=False):
        return self.fm_forward(batch_feature, discrete, o_sigmoid)
    