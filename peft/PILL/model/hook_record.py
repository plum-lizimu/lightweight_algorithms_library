from typing import Any, Tuple
import torch.nn as nn

class hook_record:

    def __init__(self, model, recorder_moudle, _adapter):
        self.record_tag = False
        self.model = model
        self.adapter_list = _adapter['adapter']
        self.method = _adapter['method']
        for name, module in model.named_modules():
            if name == recorder_moudle:
                module.register_forward_hook(self.forward_hook)
                break

    def forward_hook(self, moudle: nn.Module, inputs: Tuple, outputs: Any):
        if self.record_tag:
            adapter_out = []
            for adapter in self.adapter_list:
                adapter_out.append(adapter(outputs))
            if self.method == None:
                new_outputs = outputs + sum(adapter_out)
            else:
                new_outputs = outputs + self.method(self.model.data_pool, adapter_out)
            return new_outputs
        return outputs

    def __enter__(self):
        self.record_tag = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.record_tag = False



class data_record:

    def __init__(self, model, recorder_moudle):
        self.record_tag = False
        # self.model = model
        # self.adapter_list = _adapter['adapter']
        # self.method = _adapter['method']
        for name, module in model.named_modules():
            if name == recorder_moudle:
                print(name)
                module.register_forward_hook(self.forward_hook)
                break

    def forward_hook(self, moudle: nn.Module, inputs: Tuple, outputs: Any):
        if self.record_tag:
            print(len(outputs))
            # print(f'before: { outputs }')
            # adapter_out = []
            # for adapter in self.adapter_list:
            #     adapter_out.append(adapter(outputs))
            # if self.method == None:
            #     outputs += sum(adapter_out)
            # else:
            #     outputs += self.method(self.model.data_pool, adapter_out)
            # print(f'after: { outputs }')
        # return outputs

    def __enter__(self):
        self.record_tag = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.record_tag = False

