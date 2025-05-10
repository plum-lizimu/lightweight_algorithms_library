from .Base import Base
import torch
import torch.nn as nn

class CrossNet(Base):
 
    def __init__(self, params):
        super().__init__(params=params)
        self.layer_num = 2
        self.kernels = nn.Parameter(torch.Tensor(self.layer_num, self.feat_dim, self.feat_dim))
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, self.feat_dim, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])
 
    def cross_forward(self, batch_feature, discrete, o_sigmoid):
        feature_embedding_list = self.get_embedding(batch_feature, discrete)
        all_feature = torch.concat(feature_embedding_list, -1)
        x_0 = all_feature.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
           xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
           dot_ = xl_w + self.bias[i]  # W * xi + b
           x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
        x_l = torch.squeeze(x_l, dim=2)
        return x_l

    def forward(self, batch_feature, discrete=True, o_sigmoid=False):
        return self.cross_forward(batch_feature, discrete, o_sigmoid)