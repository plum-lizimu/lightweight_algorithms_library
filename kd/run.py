## 示例代码,可根据具体任务进行调整

import os
import sys
sys.path.append("../")

from utils import getDataloader, logger
from task_embarkation import embarkation_dict

import config
import torch

def main():
    base_config = config.base_config

    ## 加载实验设置
    experiment_config = base_config['experiment_config']
    device = torch.device(experiment_config['device'] if torch.cuda.is_available() else 'cpu')

    ## 加载数据集,获得dataloader
    dataloader_config = base_config['dataloader']
    init_params = dataloader_config['init_params']
    dataloader_moudle = getDataloader(dataset_name=dataloader_config['dataset_name'], 
                                      data_path=init_params['data_path'])
    train_dataloader = dataloader_moudle.train_dataloader(**dataloader_config['train_dataloader_params'])
    test_dataloader = dataloader_moudle.test_dataloader(**dataloader_config['test_dataloader_params'])

    ## 加载教师模型
    teacher_config = base_config['teacher_model']
    teacher = teacher_config['model']
    if teacher_config['ckpt'] is not None:
        teacher.load_state_dict(torch.load(teacher_config['ckpt']))
    teacher.to(device)

    ## 构建学生模型
    student_config = base_config['student_model']
    student = student_config['model']
    student.to(device)
    save_path = student_config['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    _logger = logger(save_path)
    
    ## 构建 optimizer 和 scheduler
    optimizer_config = experiment_config['optimizer']
    optimizer_config['params']['params'] = student.parameters()
    optimizer = optimizer_config['type'](**optimizer_config['params'])

    scheduler_config = experiment_config['scheduler']
    scheduler = None
    if scheduler_config is not None:
        scheduler_config['params']['optimizer'] = optimizer
        scheduler = scheduler_config['type'](**scheduler_config['params'])

    ## 获得蒸馏装载器
    kd_embarkation = embarkation_dict[experiment_config['embarkation']['type']](teacher, student)

    ## 预处理 losses_mapping
    losses_mapping = experiment_config['losses_mapping']
    for _, loss_setting in losses_mapping.items():
        loss_fn = loss_setting['type'](**loss_setting['init_params'])
        loss_setting.pop('type')
        loss_setting.pop('init_params')
        loss_setting['loss_fn'] = loss_fn
    
    ## 蒸馏前操作
    kd_embarkation.preKD(test_data=test_dataloader, logger=_logger) ## 蒸馏中途需要的额外的东西进行暂存

    ## 进行蒸馏操作
    kd_embarkation.train(train_data=train_dataloader, 
                         epoch=experiment_config['epoch'],
                         optimizer=optimizer,
                         scheduler=scheduler,
                         losses_mapping=losses_mapping,
                         device=device)
    
    ## 蒸馏后操作
    kd_embarkation.postKD()

if __name__ == '__main__':
    main()