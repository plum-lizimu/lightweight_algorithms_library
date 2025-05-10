import torch.nn as nn
from typing import Dict, Tuple, Union

'''
    二维卷积层对齐操作
'''
class conv2d_pipe(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]]=1, 
                 stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0, groups: int=1,
                 padding_mode: str='zeros', use_bn: bool=False, act_cfg: Dict=dict(type='ReLU', params={})) -> None:
        super().__init__()
        self.alignment = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, padding_mode=padding_mode, bias=False)
        self.use_bn = use_bn
        self.use_act = act_cfg != {}
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            assert isinstance(act_cfg, dict) and 'type' in act_cfg
            try:
                # Get the activation layer class from torch.nn using reflection
                act_class = getattr(nn, act_cfg['type'])
            except AttributeError:
                raise ValueError(f"Unsupported activation type: { act_cfg['type'] }")
            self.act = act_class(**act_cfg['params'])
    
    def forward(self, feature):
        _out = self.alignment(feature)
        if self.use_bn:
            _out = self.bn(_out)
        if self.use_act:
            _out = self.act(_out)
        return _out
    
'''
    直接转发输入特征
'''
class transmit(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, feature):
        return feature
    
    
pipe_dict = {
    'conv2d_pipe': conv2d_pipe,
    'transmit': transmit
}
        