from .Base import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DNN(Base):

    def __init__(self, params):
        super().__init__(params)
        self.hidden_dim = params['hidden_dim']
        self.use_fc = params.get('use_fc', True)
        self.all_dim = [self.feat_dim] + self.hidden_dim + [1]
        for i in range(1, len(self.hidden_dim) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dim[i-1], self.all_dim[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dim[i]))
            setattr(self, 'activation_' + str(i), nn.LeakyReLU())
        if self.use_fc:
            self.fc = nn.Linear(self.hidden_dim[-1], 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.normal_(weight, 0, 1/math.sqrt(self.dim))
    
    def dnn_forward(self, batch_feature, discrete, o_sigmoid):
        feature_embedding_list = self.get_embedding(batch_feature, discrete)
        deep_out = torch.concat(feature_embedding_list, -1)
        for i in range(1, len(self.hidden_dim) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'activation_' + str(i))(deep_out)
        if self.use_fc:
            deep_out = torch.squeeze(self.fc(deep_out))
            if o_sigmoid:
                return F.sigmoid(deep_out)
        return deep_out
    
    def forward(self, batch_feature, discrete=True, o_sigmoid=False):
        return self.dnn_forward(batch_feature, discrete, o_sigmoid)
    