'''
    任务: Cifar100数据集下,对Resnet34模型进行剪枝，剪枝率为20%
'''
import torch
import torch.nn as nn
from models import model_dict

import torch_pruning as tp

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
    "pruning_model": {
        "model": model_dict['resnet34'](num_classes=100),
        "ckpt": '/home/hyt/lightweight_algorithms_library/kd/teacher_model/resnet34-183-best_cifar100.pth',
        "save_path": '/home/hyt/lightweight_algorithms_library/experiment/pruning/cifar100_resnet34_pruning'
    },
    "experiment_config": {
        "device": "cuda:0",
        ## 剪枝后的微调次数
        "epoch": 3,
        ## 进行微调时的优化器参数
        "optimizer": {
            "type": torch.optim.SGD,
            "params": {
                "lr": 8e-3,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        },
        "scheduler": None,
        "prune_config": {
            "importance": {
                "type": "MagnitudeImportance",  ## 评估某个结构的重要性程度
                "params": {
                    "p": 2, 
                    "group_reduction": "mean"
                }
            },
            "pruner": {
                "type": "MagnitudePruner",
                "params": {
                    "global_pruning": False,  ## 是否进行全局剪枝
                    "iterative_steps": 3,     ## 剪枝迭代次数
                    "pruning_ratio": 0.2,     ## 剪枝率
                    "ignored_layers": ["fc"]  ## 不进行剪枝的层
                }
            },
        },
        "embarkation": {
            "type": "cifarPruningEmbarkation"
        },
        "losses_mapping": {
            "loss_base": {
                "type": nn.CrossEntropyLoss,
                "init_params": {},
                "weight": 1
            }
        }
    }
}