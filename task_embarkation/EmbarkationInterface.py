from abc import ABC, abstractmethod    #定义抽象类
from typing import Union, Callable
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn

'''
    蒸馏工作装载器
'''
class KDEmbarkationInterface(ABC):

    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student

    '''
        做蒸馏前准备工作
    '''
    @abstractmethod
    def preKD(self, **kwargs):
        pass

    '''
        做蒸馏训练
    '''
    @abstractmethod
    def train(self, 
              train_data: Union[Dataset, DataLoader], 
              epoch: int, 
              optimizer: Optimizer = None, 
              scheduler: _LRScheduler = None, 
              losses_mapping: dict = None,
              device: torch.device = 'cpu',
              **kwargs):
        pass
    
    '''
        验证测试
    '''
    @abstractmethod
    def validate(self, 
                 val_data: Union[Dataset, DataLoader],
                 criterion: Callable = None,
                 device: torch.device = 'cpu',
                 **kwargs):
        pass

    '''
        蒸馏收尾工作
    '''
    @abstractmethod
    def postKD(self, **kwargs):
        pass
    
    '''
        模型存储
    '''
    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        pass

    
'''
    剪枝模块装载器
'''
class PruneEmbarkationInterface(ABC):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    '''
        做剪枝前准备工作
    '''
    @abstractmethod
    def prePruning(self, importance: nn.Module = None, pruner: nn.Module = None, init_config: dict = None, **kwargs):
        pass

    '''
        剪枝,并进行微调
    '''
    @abstractmethod
    def train(self, 
              train_data: Union[Dataset, DataLoader], 
              iterative_steps: int, # 剪枝次数
              epoch: int,           # 微调次数
              optimizer: Optimizer = None, 
              scheduler: _LRScheduler = None, 
              losses_mapping: dict = None,
              device: torch.device = 'cpu',
              **kwargs):
        pass
    
    '''
        验证测试
    '''
    @abstractmethod
    def validate(self, 
                 val_data: Union[Dataset, DataLoader],
                 criterion: Callable = None,
                 device: torch.device = 'cpu',
                 **kwargs):
        pass

    '''
        剪枝收尾工作
    '''
    @abstractmethod
    def postPruning(self, **kwargs):
        pass
    
    '''
        模型存储
    '''
    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        pass

    '''
        得到样例输入(example_inputs)
    '''
    @abstractmethod
    def get_example_input(self, data: Union[Dataset, DataLoader], **kwargs):
        pass


'''
    Peft模块装载器
'''
class PeftEmbarkationInterface(ABC):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    '''
        做peft前准备工作
    '''
    @abstractmethod
    def prePeft(self, **kwargs):
        pass

    '''
        peft训练
    '''
    @abstractmethod
    def train(self, 
              train_data: Union[Dataset, DataLoader], 
              epoch: int, 
              optimizer: Optimizer = None, 
              scheduler: _LRScheduler = None, 
              losses_mapping: dict = None,
              device: torch.device = 'cpu',
              **kwargs):
        pass
    
    '''
        验证测试
    '''
    @abstractmethod
    def validate(self, 
                 val_data: Union[Dataset, DataLoader],
                 criterion: Callable = None,
                 device: torch.device = 'cpu',
                 **kwargs):
        pass

    '''
        peft收尾工作
    '''
    @abstractmethod
    def postPeft(self, **kwargs):
        pass
    
    '''
        模型存储
    '''
    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        pass


'''
    量化感知训练(QAT)模块装载器
'''
class QTAEmbarkationInterface(ABC):
    def __init__(self, model: nn.Module, test_data: Union[Dataset, DataLoader], logger: Callable) -> None:
        super().__init__()
        self.model = model
        self.test_data = test_data
        self.logger = logger

    '''
        量化前的准备,包括量化感知训练(QAT)的设置
    '''
    @abstractmethod
    def preQuantization(self, **kwargs):
        pass

    '''
        执行量化感知训练(QAT)
    '''
    @abstractmethod
    def train_with_quantization(self, 
                                train_data: Union[Dataset, DataLoader], 
                                epoch: int, 
                                optimizer: Optimizer = None, 
                                scheduler: _LRScheduler = None, 
                                losses_mapping: dict = None, 
                                device: torch.device = 'cpu',
                                **kwargs):
        pass

    '''
        量化后的收尾工作，包括将模型转换为量化推理格式
    '''
    @abstractmethod
    def postQuantization(self, **kwargs):
        pass

    '''
        验证量化后模型的性能
    '''
    @abstractmethod
    def validate(self, 
                 val_data: Union[Dataset, DataLoader], 
                 device: torch.device = 'cpu',
                 **kwargs):
        pass

    '''
        保存量化后的模型
    '''
    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        pass

    '''
        获取样例输入（用于计算模型的参数和MACs）
    '''
    @abstractmethod
    def get_example_input(self, data: Union[Dataset, DataLoader], device: torch.device, **kwargs):
        pass

'''
    动态量化模块装载器
'''
class DYQEmbarkationInterface(ABC):
    def __init__(self, model: nn.Module, test_data: Union[Dataset, DataLoader], logger: Callable) -> None:
        """
        动态量化模型的接口定义。

        参数：
        - model: nn.Module
            要进行动态量化的神经网络模型。
        - test_data: Dataset | DataLoader
            用于验证模型性能的数据集或数据加载器。
        - logger: Callable
            用于记录日志的函数。
        """
        super().__init__()
        self.model = model
        self.test_data = test_data
        self.logger = logger

    @abstractmethod
    def preQuantization(self, **kwargs):
        """
        动态量化前的准备工作，包括验证模型量化前的性能。

        参数：
        - kwargs: dict
            其他可选参数。
        """
        pass

    @abstractmethod
    def apply_dynamic_quantization(self):
        """
        执行动态量化，将模型进行低精度（如 qint8）量化。
        """
        pass

    @abstractmethod
    def postQuantization(self, **kwargs):
        """
        动态量化后的收尾工作，包括验证量化后的性能并记录结果。

        参数：
        - kwargs: dict
            其他可选参数。
        """
        pass

    @abstractmethod
    def validate(self, val_data: Union[Dataset, DataLoader], device: torch.device = 'cpu'):
        """
        验证模型的性能。

        参数：
        - val_data: Dataset | DataLoader
            用于验证模型的数据集或数据加载器。
        - device: torch.device
            运行验证的设备（默认为 'cpu'）。

        返回：
        - acc_1: float
            模型的 Top-1 准确率。
        - acc_5: float
            模型的 Top-5 准确率。
        """
        pass

    @abstractmethod
    def save_model(self, save_path: str, **kwargs):
        """
        保存动态量化后的模型。

        参数：
        - save_path: str
            保存模型的路径。
        - kwargs: dict
            其他可选参数。
        """
        pass

    @abstractmethod
    def get_example_input(self, data: Union[Dataset, DataLoader], device: torch.device, **kwargs):
        """
        提供模型运行的示例输入（用于计算参数量和 MACs）。

        参数：
        - data: Dataset | DataLoader
            数据集或数据加载器，用于提取示例输入。
        - device: torch.device
            设备（如 'cpu' 或 'cuda'），用于存放示例输入。
        - kwargs: dict
            其他可选参数。

        返回：
        - example_input: torch.Tensor
            示例输入张量。
        """
        pass