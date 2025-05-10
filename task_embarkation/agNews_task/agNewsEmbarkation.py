import os
import torch
from task_embarkation.EmbarkationInterface import PeftEmbarkationInterface
from peft import LoraConfig, PrefixTuningConfig, PeftConfig, get_peft_model, get_peft_model_state_dict
from transformers import Trainer

'''
    采用transformers库中的相关方法实现
'''
class agNewsEmbarkation(PeftEmbarkationInterface):

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.trainer = None

    def load_model(self, path=None, **kwargs):
        assert os.path.exists(path), f"not found { path } file."
        self.model.load_state_dict(torch.load(path))

    def save_model(self, save_path: str, **kwargs):
        assert os.path.exists(save_path), f"not found { save_path } file."
        if 'save_peft' in kwargs and kwargs['save_peft']:
            torch.save(get_peft_model_state_dict(self.model), f"{ save_path }/lora_parameters.pth")
        else:
            torch.save(self.model.state_dict(), f"{ save_path }/model.pth")
    
    def prePeft(self, **kwargs):
        type = kwargs['type']
        if type == 'lora':
            _peft_config = LoraConfig(**kwargs['params'])
        elif type == 'prefixTuning':
            _peft_config = PrefixTuningConfig(**kwargs['params'])
        self.model = get_peft_model(self.model, _peft_config)
    
    def train(self, dataloader=None, **kwargs):
        if self.trainer is None:
            self.trainer = Trainer(
                model=self.model,
                **kwargs 
            )
        self.trainer.train()
    
    def validate(self, val_data, criterion = None, device = 'cpu', **kwargs):
        return super().validate(val_data, criterion, device, **kwargs)
    
    def postPeft(self, **kwargs):
        return super().postPeft(**kwargs)