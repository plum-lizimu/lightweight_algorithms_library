import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from dataloader.DataloaderInterface import DataloaderInterface

class Dataloader(DataloaderInterface):

    class _InnerDataset(Dataset):  # 内部类实现
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return {
                'user':  self.data.iloc[index].user_id,
                'item':  self.data.iloc[index].item_id,
                'label': self.data.iloc[index].score
            }
    
    @staticmethod
    def _collate_fn(batch_data):  # 静态方法处理批数据
        batch_users = []
        batch_items = []
        batch_labels = []

        for data in batch_data:
            batch_users.append(data['user'])
            batch_items.append(data['item'])
            batch_labels.append(data['label'])

        return {
            'user': torch.LongTensor(batch_users),
            'item': torch.LongTensor(batch_items),
            'label': torch.FloatTensor(batch_labels)
        }

    def load_dateset(self, type, **kwargs):
        dataset_folder = os.path.join(self.data_path, 'ml_1m')
        data = pd.read_csv(os.path.join(dataset_folder, f'{ type }.csv'))
        dataset = self._InnerDataset(data)
        user_num = data.user_id.max() + 1
        item_num = data.item_id.max() + 1
        return dataset, user_num, item_num

    def load_dataloader(self, type, **kwargs):
        dataset, user_num, item_num = self.load_dateset(type)
        dataloader = DataLoader(dataset, 
                                batch_size=kwargs['batch_size'],
                                num_workers=kwargs['num_workers'], 
                                shuffle=kwargs['shuffle'], 
                                collate_fn=self._collate_fn, 
                                drop_last=True)
        return dataloader, user_num, item_num
    
    def train_dataset(self, **kwargs):
        return self.load_dateset('train')
    
    def val_dataset(self, **kwargs):
        return self.load_dateset('val')
    
    def test_dataset(self, **kwargs):
        return self.load_dateset('test')

    def train_dataloader(self, **kwargs):
        return self.load_dataloader('train', **kwargs)
    
    def test_dataloader(self, **kwargs):
        return self.load_dataloader('test', **kwargs)
    
    def val_dataloader(self, **kwargs):
        return self.load_dataloader('val', **kwargs)