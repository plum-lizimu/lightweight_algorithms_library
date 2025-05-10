import torch
import torch.nn as nn
import torch.nn.functional as F

from .CrossNet import CrossNet
from .DNN import DNN

class DCN(CrossNet, DNN):

    def __init__(self, params):
        params['use_fc'] = False
        super().__init__(params=params)

        self.fc = nn.Linear(self.feat_dim + params['hidden_dim'][-1], 1)

    def dcn_forward(self, batch_feature, discrete, o_sigmoid):
        dnn_out = self.dnn_forward(batch_feature, discrete, o_sigmoid=False)
        cross_out = self.cross_forward(batch_feature, discrete, o_sigmoid=False)
        concat_out = torch.cat([cross_out, dnn_out], dim=-1)
        result = torch.squeeze(self.fc(concat_out))
        if o_sigmoid:
            return F.sigmoid(result)
        return result

    def forward(self, batch_feature, discrete=True, o_sigmoid=False):
        return self.dcn_forward(batch_feature, discrete, o_sigmoid) 
