from typing import Optional
import torch
import torch.nn as nn

from model.llm_model import TransformerBlock
from model.llm_model import ModelArgs
from model.adapters import adapter_dict
from model.hook_record import hook_record, data_record

# class AdapterTransformerBlock(TransformerBlock):

#     def __init__(self, layer_id: int, args: ModelArgs, config: dict, parent: nn.Module, adapter_record: dict):
#         super().__init__(layer_id, args)
    
#         # for (name_p, param_p), (name_c, param_c) in zip(parent.named_parameters(), self.named_parameters()):
#         #     if name_p == name_c:
#         #         param_c.data.copy_(param_p.data)
        
#         for adapter_config in config['Adapter_list']:
#             adapter = adapter_dict[adapter_config['type']](**adapter_config['params'])
#             setattr(self, adapter_config['adapter_name'], adapter)
#             layer_name = adapter_config['pre_layer']
#             if layer_name not in adapter_record:
#                 adapter_record[layer_name] = {'adapter': [], 'method': None}
#             adapter_record[layer_name]['adapter'].append(adapter)
#             method = config['fun_list'][adapter_config['operation']]
#             adapter_record[layer_name]['method'] = method
        
#         self.data_pool = {}
        
#     def forward(self, x: torch.Tensor, 
#                 start_pos: int, 
#                 freqs_cis: torch.Tensor, 
#                 mask: Optional[torch.Tensor], 
#                 image_position,
#                 ):
#         x_norm = self.attention_norm(x)
#         # x_a = torch.where(image_position, self.adapter_attn_v(x_norm), self.adapter_attn_l(x_norm))
#         # loss = F.mse_loss(x_v, x_l)

#         x_a = self.adapter_attn(x_norm)
#         h = self.attention.forward(x_a+x_norm, start_pos, freqs_cis, mask, image_position) + x
#         # h = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, image_position) + x
#         # h += x
#         h_norm = self.ffn_norm(h)
#         h_v, h_l = self.vis_adapter_ffn(h_norm), self.txt_adapter_ffn(h_norm)
#         # h_l = self.txt_adapter_ffn(h_norm)
#         h_a = torch.where(image_position, h_v, h_l)

#         # h_a = h_l
#         h_f = self.feed_forward.forward(h_norm + h_a) + h
#         # h_f = self.feed_forward.forward(h_norm) + h + h_a

#         return h_f

# class AdapterTransformerBlock(nn.Module):

#     def __init__(self, Parent: nn.Module, ):
#         super().__init__()
#         self.model = Parent

def forward_adapter_pill_hook(self, x: torch.Tensor, 
                               start_pos: int, 
                               freqs_cis: torch.Tensor, 
                               mask: Optional[torch.Tensor], 
                               image_position):
    self.data_pool['image_position'] = image_position
    return self._original_forward(x, start_pos, freqs_cis, mask, image_position)


def forward_adapter_pill(self, x: torch.Tensor, 
                               start_pos: int, 
                               freqs_cis: torch.Tensor, 
                               mask: Optional[torch.Tensor], 
                               image_position):
    x_norm = self.attention_norm(x)
    x_a = self.adapter_attn(x_norm)

    h = self.attention.forward(x_a+x_norm, start_pos, freqs_cis, mask, image_position) + x
    h_norm = self.ffn_norm(h)
    
    h_v, h_l = self.vis_adapter_ffn(h_norm), self.txt_adapter_ffn(h_norm)
    h_a = torch.where(image_position, h_v, h_l)
    h_f = self.feed_forward.forward(h_norm + h_a) + h
    return h_f


### 设计一种hook机制,在forward时判断该层是否被注册到,若被注册到,自动调用附属于该层adapter,
### 若附属于该层的adapter只有一个,则值直接累加;否则,查看该是否带有其他处理方法,对adapter下的输出结果处理之后,再相加
def peft_adapter(model, config, **kwargs):
    blocks = getattr(model, config['target_part'])
    block_num = config['block_num']
    ModelArgs = kwargs['model_args']

    hook_list = []
    for block_id in range(block_num):
        block = blocks[block_id]
        ## 替换block
        adapter_record = {}
        # model.layers[block_id] = AdapterTransformerBlock(block_id, ModelArgs, config, block, adapter_record)
        # del block
        for adapter_config in config['Adapter_list']:
            adapter = adapter_dict[adapter_config['type']](**adapter_config['params'])
            setattr(block, adapter_config['adapter_name'], adapter)
            layer_name = adapter_config['pre_layer']
            if layer_name not in adapter_record:
                adapter_record[layer_name] = {'adapter': [], 'method': None}
            adapter_record[layer_name]['adapter'].append(adapter)
            method = config['fun_list'][adapter_config['operation']]
            adapter_record[layer_name]['method'] = method
        setattr(block, 'data_pool', {})
        
        # # print(block)
        for module, _adapter in adapter_record.items():
            # module = '.'.join([config['target_part'], str(block_id), module])
            hook_list.append(hook_record(block, module, _adapter))
        
        block._original_forward = block.forward
        block.forward = forward_adapter_pill_hook.__get__(block, block.__class__)
        # block.forward = forward_adapter_pill.__get__(block, block.__class__)
        

        # print(adapter_record)
        # exit(0)
        # hook_list.append(data_record(model, f'layers.{block_id}.attention'))
        # print(len(hook_list))
        # exit(0)
    
    return model, hook_list


# def peft_adapter_block(block, config):
#     hook_list = []
#     adapter_record = {}
#     model = AdapterTransformerBlock(0, ModelArgs, config, block, adapter_record)
   
#     for module, _adapter in adapter_record.items():
#         hook_list.append(hook_record(model, module, _adapter))

#     return model, hook_list