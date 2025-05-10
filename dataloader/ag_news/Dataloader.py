import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from interface.DataloaderInterface import DataloaderInterface
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

class Dataloader(DataloaderInterface):

    def __init__(self, dataset_name, dataset_path, **kwargs):
        super().__init__(dataset_name, dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name'])
        self.dataset = load_dataset(self.dataset_path)

    def collate_fn(self, batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']).squeeze(0) for item in batch])
        token_type_ids = torch.stack([torch.tensor(item['token_type_ids']).squeeze(0) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']).squeeze(0) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 
                'attention_mask': attention_mask, 'label': labels}

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors="pt")
    
    def train_dataloader(self, **kwargs):
        train_dataset = self.dataset['train']
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        if kwargs['return_dataset']:
            return train_dataset
        else:
            kwargs.pop('return_dataset')
            kwargs['collate_fn'] = self.collate_fn
            train_dataloader = DataLoader(train_dataset, **kwargs)
            return train_dataloader
    
    def test_dataloader(self, **kwargs):
        test_dataset = self.dataset['test']
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        if kwargs['return_dataset']:
            return test_dataset
        else:
            kwargs.pop('return_dataset')
            kwargs['collate_fn'] = self.collate_fn
            test_dataloader = DataLoader(test_dataset, **kwargs)
            return test_dataloader
    
    def val_dataloader(self, **kwargs):
        return super().val_dataloader(**kwargs)

    # def get_dataset(self):
    #     return {
    #         'train': self.dataset['train'].map(self.tokenize_function, batched=True),
    #         'test': self.dataset['test'].map(self.tokenize_function, batched=True)
    #     }
