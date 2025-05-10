from .DNN import DNN
from .FM import FM
import torch.nn.functional as F


class DeepFM(FM, DNN):

    def __init__(self, params):
        super().__init__(params)

    def deepfm_forward(self, batch_feature, discrete, o_sigmoid):
        result_1 = self.fm_forward(batch_feature, discrete, o_sigmoid=False)
        result_2 = self.dnn_forward(batch_feature, discrete, o_sigmoid=False)
        result = result_1 + result_2
        if o_sigmoid:
            return F.sigmoid(result)
        return result

    def forward(self, batch_feature, discrete=True, o_sigmoid=False):
        return self.deepfm_forward(batch_feature, discrete, o_sigmoid)
    