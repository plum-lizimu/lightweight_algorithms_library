import torch.nn as nn
import torch.nn.functional as F
import math

class Gating(nn.Module):
    def __init__(self, teacher_num, gen_batch_size, polar_t=7., dp_rate=.2):
        super().__init__()
        self.polar_t = polar_t
        dim = teacher_num * gen_batch_size

        self.net = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(dim*4, dim*4),
            nn.ReLU(),
            nn.Dropout(p=.6),

            nn.Linear(dim*4, dim*2),
            nn.ReLU(),
            nn.Dropout(p=.6),

            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Dropout(p=.6),

            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(p=.6),
            
            nn.Linear(dim//2, dim // 4),
            nn.ReLU(),
            nn.Dropout(p=.5),
            
            nn.Linear(dim // 4, teacher_num)
        )

        self.softmax_layer = nn.Softmax(dim=0)
        self.dropout_layer = nn.Dropout(p=dp_rate)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.net[-1].weight.data.fill_(1.0)
        self.net[-1].bias.data.fill_(0.0)
    
    def forward(self, x, mask=True):
        x = x.view(-1)
        out = self.net(x)
        out = self.softmax_layer(out * math.exp(-self.polar_t))
        if mask:
            out = self.dropout_layer(out)
            out = out / (out.sum() + 1e-15)
        return out
    