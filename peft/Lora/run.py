import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append("../../")

from utils import logger
from task_embarkation import embarkation_dict
from models import model_dict

import json
import torch
import torch.nn as nn
import importlib
import torch.optim as optim
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
from transformers import TrainingArguments

def main():

    ### 读取配置
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
        embarkation_base = config['base_config']['embarkation_base']
        dataloader_config = config['dataloader']
        model_config = config['model']
        peft_config = config['peft_config']
        experiment_config = config['experiment_settings']
    
    ### 加载数据集
    dataset_name = dataloader_config['init_params']['dataset_name']
    dataloader_import_path = f'dataloader.{dataset_name}.Dataloader'
    dataloader_module = importlib.import_module(dataloader_import_path)

    dataloader_base = dataloader_module.Dataloader(**dataloader_config['init_params'])
    train_dataloader = dataloader_base.train_dataloader(**dataloader_config['train_params'])
    test_dataloader = dataloader_base.test_dataloader(**dataloader_config['test_params'])

    ### 加载实验设置
    device = torch.device(experiment_config['device'])
    args = TrainingArguments(**experiment_config['TrainingArguments'])

    ### 加载模型
    model_name = model_config['model_name']
    model_params = model_config['params']
    from_pretrained = model_config['from_pretrained']
    ckpt = model_config['ckpt']
    if from_pretrained:
        model = model_dict[model_name].from_pretrained(**model_params)
    else:
        model = model_dict[model_name](**model_params)
    model.to(device)

    ### 添加 LoRA 层
    if peft_config['type'] == 'lora':
        lora_params = peft_config['params']
        lora_config = LoraConfig(**lora_params)
        model = get_peft_model(model, lora_config)

    ### 初始化模型
    embarkation = embarkation_dict['agNewsEmbarkation'](model)
    if ckpt is not None:
        embarkation.load_model(ckpt)

    embarkation.prePeft(**peft_config)

    embarkation.train(args=args, train_dataset=train_dataloader, eval_dataset=test_dataloader)

    embarkation.postPeft()

    embarkation.save_model(path='./experiment/record_expr1', save_peft=True)

if __name__ == '__main__':
    main()
