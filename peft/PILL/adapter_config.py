import torch

adapter_config = {
    'target_part': 'layers',
    'Adapter_list': [
        {
            'adapter_name': 'adapter_attn',		## Adapter 的命名
            'type': 'GLUAdapter',				## 注入的Adapter类型
            'params': {
                'args': None                    ## Adapter初始化参数, 程序中确定
            },
            'pre_layer': 'attention_norm',      ## 指明 Adapter 上一层
            'operation': 'default'				## Adapter 输出后的操作
        },
        {
            'adapter_name': 'vis_adapter_ffn',
            'type': 'GLUAdapter',
            'params': {
                'args': None
            },
            'pre_layer': 'ffn_norm',
            'operation': 'fun_1'
        },
        {
            'adapter_name': 'txt_adapter_ffn',
            'type': 'GLUAdapter',
            'params': {
                'args': None
            },
            'pre_layer': 'ffn_norm',
            'operation': 'fun_1'
        }
    ],
    'fun_list': {
        'default': None,
        'fun_1': 
            lambda data_pool, adapter_out_list:
                torch.where(data_pool['image_position'], adapter_out_list[0], adapter_out_list[1])
    },
    # 'data_pool': ['image_position']
}