'''
    上下文管理器，用于记录模型的中间层输出
'''
from typing import Any, Tuple
import torch.nn as nn

class feature_map_recorder:

    def __init__(self, model, recorder_moudle):
        self.record_tag = False
        self.data = None
        for name, module in model.named_modules():
            if name == recorder_moudle:
                module.register_forward_hook(self.forward_hook)
                break

    def forward_hook(self, moudle: nn.Module, inputs: Tuple, outputs: Any):
        if self.record_tag:
            self.data = outputs

    def __enter__(self):
        self.record_tag = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.record_tag = False

    def get_record_data(self):
        if self.record_tag:
            return self.data
        return None
