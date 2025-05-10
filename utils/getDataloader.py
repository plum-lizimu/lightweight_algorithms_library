import os
import sys
import importlib

def getDataloader(dataset_name: str, data_path: str = None):
    ## data_path为空, 数据集将从项目内部文件夹data导入
    if data_path is None:
        data_path = os.path.join('..', 'data', dataset_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The path { data_path } dose not exist.")
    dataloader_path = f'dataloader.{ dataset_name }.Dataloader'
    dataloader_moudle = importlib.import_module(dataloader_path)
    return dataloader_moudle.Dataloader(data_path)