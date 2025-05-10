import json
import torch
import sys
sys.path.append("../")
from task_embarkation.cifar_task.cifarEmbarkation import cifarKDEmbarkation, cifarPruningEmbarkation, cifarQATEmbarkation, cifarDynamicQuantizationEmbarkation

import importlib
import os

from utils import getDataloader, logger
from models import model_dict
# import torch_pruning as tp

class EmbarkationManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def load_model(self, model_name, model_path):
        model = model_dict[model_name](num_classes=100)
        ckpt = torch.load(model_path)  # 根据用户给定的模型路径加载模型
        model.load_state_dict(ckpt)
        return model
    
    def run_task(self):
        task_type = self.config["task"]  # 判断任务类型是蒸馏、剪枝还是量化

        ## 加载dataloader
        self.train_dataloader, self.test_dataloader = self.get_dataloader(self.config)
        
        ## 加载logger
        save_path = self.config['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._logger = logger(self.config['save_path'])

        if task_type == "qat":
            self.run_qat()
        elif task_type == "dyq":
            self.run_dyq()
    
    ## 获取数据集加载器
    def get_dataloader(self, config):
        dataloader_config = config['dataloader']
        init_params = dataloader_config['init_params']
        dataloader_module = getDataloader(dataset_name=dataloader_config['dataset_name'], 
                                          data_path=init_params['data_path'])        
        train_dataloader = dataloader_module.train_dataloader(**dataloader_config['train_dataloader_params'])
        test_dataloader = dataloader_module.test_dataloader(**dataloader_config['test_dataloader_params'])
        return train_dataloader, test_dataloader
    
    def run_qat(self):
        torch.backends.quantized.engine = 'fbgemm'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        quant_model = self.load_model(self.config['model']['model_name'], self.config['model']['quantized_model_path'])
        quant_model.to(device)  # 将模型移至指定设备

        # 初始化量化任务
        quant_task = cifarQATEmbarkation(model=quant_model, test_data=self.test_dataloader, logger=self._logger)
        quant_task.preQuantization(device=device)  # 量化前的准备

        # 训练量化模型（QAT）
        quant_task.train_with_quantization(
            train_data=self.train_dataloader,
            epoch=self.config['train_params']['epochs'],
            optimizer=torch.optim.SGD(quant_model.parameters(), lr=self.config['train_params']['learning_rate']),
            scheduler=None,
            losses_mapping=self.config['losses_mapping'],
            device=device
        )

        # 保存量化后的模型
        quant_task.postQuantization(save_path=self.config['save_path'])  # 保存量化模型
    
    def run_dyq(self):
        torch.backends.quantized.engine = 'fbgemm'  # 设置量化后端
        device = torch.device('cpu')  # 动态量化强制使用CPU

        # 加载原始模型
        model = self.load_model(self.config['model']['model_name'], self.config['model']['quantized_model_path'])
        model.to(device)

        # 初始化动态量化任务
        quant_task = cifarDynamicQuantizationEmbarkation(
            model=model,
            test_data=self.test_dataloader,
            logger=self._logger
        )
        
        # 执行量化流程
        quant_task.preQuantization()
        quant_task.apply_dynamic_quantization()
        quant_task.postQuantization()
        quant_task.save_model(self.config['save_path'])
