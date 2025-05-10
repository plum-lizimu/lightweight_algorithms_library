from abc import ABC, abstractmethod

class DataloaderInterface(ABC):
    
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
    
    @abstractmethod
    def train_dataset(self, **kwargs):
        """获取训练集"""
        pass

    @abstractmethod
    def val_dataset(self, **kwargs):
        """获取验证集"""
        pass

    @abstractmethod
    def test_dataset(self, **kwargs):
        """获取测试集"""
        pass

    @abstractmethod
    def train_dataloader(self, **kwargs):
        """获取训练Dataloader"""
        pass

    @abstractmethod
    def val_dataloader(self, **kwargs):
        """获取验证Dataloader"""
        pass

    @abstractmethod
    def test_dataloader(self, **kwargs):
        """获取测试Dataloader"""
        pass
