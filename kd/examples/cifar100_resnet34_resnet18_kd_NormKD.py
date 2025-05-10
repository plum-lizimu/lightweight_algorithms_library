'''
    任务: Cifar100数据集下,使用常规logits蒸馏方法,由Resnet34蒸馏获得Resnet18
'''

import torch
import torch.nn as nn
from models import model_dict
from kd.kd_losses import algorithm_dict

dataloader_params = {
    "mean": [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    "std":  [0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
    "batch_size": 16,
    "num_workers": 2,
    "shuffle": True
}

base_config = {
    "dataloader": {
        "dataset_name": "cifar_100",
        "init_params": {
            # "data_path": "../data/cifar_100"
            "data_path": None   # 从项目默认data路径获取数据集
        },
        "train_dataloader_params": dataloader_params,
        "test_dataloader_params": dataloader_params,
        "val_dataloader": None
    },
    "teacher_model": {
        "model": model_dict['resnet34'](num_classes=100),
        "ckpt": "./teacher_model/resnet34-183-best_cifar100.pth"
    },
    "student_model": {
        "model": model_dict['resnet18'](num_classes=100),
        "save_path": "../experiment/kd/cifar100_resnet34_resnet18_kd_NKD"
    },
    "experiment_config": {
        "device": "cuda:0",
        "epoch": 200,
        "optimizer": {
            "type": torch.optim.SGD,
            "params": {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        },
        "scheduler": {
            "type": torch.optim.lr_scheduler.MultiStepLR,
            "params": {
                "milestones": [60, 120, 160],
                "gamma": 0.2
            }
        },
        "embarkation": {
            "type": "cifarKDEmbarkation"
        },
        "losses_mapping": {
            "loss_kd": {
                "type": algorithm_dict['NormKD'],
                "init_params": {  ## 初始化时传入的参数                    
                    "gamma": 2,
                    "temperature": 1.5 ## 蒸馏温度
                },
                "params": {       ## forward时需要传入的参数
                    "logits_student": "student",  ## 学生的logits输出
                    "logits_teacher": "teacher",  ## 教师的logits输出
                    # "label": "labels"
                },
                "weight": 0.7     ## 在总的loss中的占比  
            },
            "loss_base": {
                "type": nn.CrossEntropyLoss,
                "init_params": {},
                "params": {       ## forward时需要传入的参数
                    "out_s": "student",  ## 学生的logits输出
                    "labels": "labels"   ## 参与监督的样本标签
                },
                "weight": 1
            }
        }
    }
}