from pyclbr import Class
from typing import Any, Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from task_embarkation.EmbarkationInterface import KDEmbarkationInterface, PruneEmbarkationInterface, QTAEmbarkationInterface, DYQEmbarkationInterface
from tqdm import tqdm

import os
import torch
import torch.nn as nn

import sys
sys.path.append("../../")
from utils import count_ops_and_params
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert, default_qconfig, quantize_dynamic

#### base function

def base_train(model: nn.Module, 
          train_data: Dataset | DataLoader, 
          device: torch.device = 'cpu', 
          epoch: int = 1,
          losses_mapping: dict = None,
          loss_base: nn.Module = nn.CrossEntropyLoss(),
          optimizer: Optimizer = None,
          scheduler: _LRScheduler = None
         ):
    model.train()

    weight = losses_mapping['loss_base']['weight']
    loss_fn = losses_mapping['loss_base']['loss_fn']

    loss_list = []
    for e in range(epoch):
        loss = 0
        loop = tqdm(train_data, leave=True)
        for _, (data, label) in enumerate(loop):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            lss = weight * loss_fn(out, label)
            loss += lss.item()
            lss.backward()
            optimizer.step()
            loop.set_description(f'[{ e + 1 } / { epoch }]')
            loop.set_postfix(loss=lss.item())
        loss /= len(train_data)
        loss_list.append(loss)
        if scheduler:
            scheduler.step()
    return loss_list

def base_validate(model: nn.Module, val_data: Dataset | DataLoader, device: torch.device = 'cpu'):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    sample_count = 0
    with torch.no_grad():
        for data, label in val_data:
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, top1 = output.topk(1, 1, True, True)
            _, top5 = output.topk(5, 1, True, True)
            correct_top1 += (top1 == label.view(-1, 1)).sum().item()
            correct_top5 += (top5 == label.view(-1, 1)).sum().item()
            sample_count += label.size(0)
    acc_1 = correct_top1 / sample_count
    acc_5 = correct_top5 / sample_count
    return acc_1, acc_5

def base_save_model(model: nn.Module, save_path: str, tag: str, **kwargs):
    save_point = os.path.join(save_path, f"{ tag }.pth")
    torch.save(model.state_dict(), save_point)


# 定义量化包装模型类
class quantModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quantStub = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequantStub = torch.ao.quantization.DeQuantStub()
    
    def add(self, residual_function, shortcut, x):
        x = self.quantStub(x)
        t = residual_function(x)
        h = shortcut(x)
        t = self.dequantStub(t)
        h = self.dequantStub(h)
        x = t + h
        return x

    def forward(self, x):
        x = self.quantStub(x)
        x = self.model.conv1(x)
        x = self.dequantStub(x)

        iter_list = [3, 4, 6, 3]  # 示例的层次结构
        bias = 2
        for layer_id in range(len(iter_list)):
            conv_layer = getattr(self.model, f'conv{ layer_id + bias }_x')
            for add_part in range(iter_list[layer_id]):
                x = self.add(conv_layer[add_part].residual_function, conv_layer[add_part].shortcut, x)

        x = self.model.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.quantStub(x)
        x = self.model.fc(x)
        x = self.dequantStub(x)

        return x

# 定义 QAT 准备函数
def prepare_for_qat(model):
    model.train()
    model.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
        weight=torch.ao.quantization.default_weight_observer.with_args(qscheme=torch.per_tensor_affine)
    )
    prepare_qat(model, inplace=True)
    return model

#######################

class cifarKDEmbarkation(KDEmbarkationInterface):
    
    def preKD(self, **kwargs):
        self.test_data = kwargs['test_data']
        self.logger = kwargs['logger']
        self.teacher.eval()
    
    def train(self, 
              train_data: Dataset | DataLoader, 
              epoch: int, 
              optimizer: Optimizer = None, 
              scheduler: _LRScheduler = None, 
              losses_mapping: dict = None, 
              device: torch.device = 'cpu',
              **kwargs):
        
        best_acc = 0
        for e in range(epoch):
            loop = tqdm(train_data, leave=True)
            self.student.train()
            loss_epoch = 0
            for _, (data, label) in enumerate(loop):
                optimizer.zero_grad()
                data, label = data.to(device), label.to(device)
                with torch.no_grad():
                    t_logit = self.teacher(data).detach()
                s_logit = self.student(data)

                loss = 0
                for _, loss_setting in losses_mapping.items():
                    forward_params = []
                    _forward_params = loss_setting['params']
                    for _, param in _forward_params.items():
                        if param == 'student':
                            o = s_logit
                        elif param == 'teacher':
                            o = t_logit
                        elif param == 'labels':
                            o = label
                        forward_params.append(o)
                    loss += loss_setting['weight'] * loss_setting['loss_fn'](*forward_params)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                loop.set_description(f'[{ e + 1 } / { epoch }]')
                loop.set_postfix(loss=loss.item())
            loss_epoch /= len(train_data)
            self.logger(f"[{ e + 1 }/{ epoch }] train_loss: { loss_epoch }")
            if scheduler is not None:
                scheduler.step()

            ## 训练完一个epoch进行验证
            acc_1, acc_5 = self.validate(self.test_data, device=device)
            if acc_1 > best_acc:
                self.save_model(self.logger.save_path)
                best_acc = acc_1
            self.logger(f"[{ e + 1 }/{ epoch }] acc@1: { acc_1 }, acc@5: { acc_5 }, best acc: { best_acc }")
            self.logger("---------------------------")

    def validate(self, val_data: Dataset | DataLoader, criterion: Callable[..., Any] = None, device: torch.device = 'cpu', **kwargs):
        return base_validate(self.student, val_data=val_data, device=device)
    
    def postKD(self, **kwargs):
        return super().postKD(**kwargs)
    
    def save_model(self, save_path: str, **kwargs):
        base_save_model(self.student, save_pat=save_path, tag="student")
        

class cifarPruningEmbarkation(PruneEmbarkationInterface):
    
    def prePruning(self, importance: nn.Module = None, pruner: nn.Module = None, prune_config: dict = None, **kwargs):

        from torch_pruning.utils.op_counter import count_ops_and_params

        self.test_data = kwargs['test_data']
        self.logger = kwargs['logger']
        
        ## 获取剪枝器参数
        importance_config = prune_config['importance']
        pruner_config = prune_config['pruner']

        import torch_pruning as tp
        pruner_config['params']['importance'] = getattr(tp.importance, importance_config['type'])(**importance_config['params'])
        
        ignored_layers_name = pruner_config['params']['ignored_layers']
        ignored_layers = []
        for name, module in self.model.named_modules():
            if name in ignored_layers_name:
                ignored_layers.append(module)
        pruner_config['params']['ignored_layers'] = ignored_layers
        pruner_config['params']['model'] = self.model
        
        ## 得到样例输入
        pruner_config['params']['example_inputs'] = self.get_example_input(data=kwargs['data'], device=kwargs['device'])

        self.pruner = getattr(tp.pruner, pruner_config['type'])(**pruner_config['params'])


    def train(self, 
              train_data: Dataset | DataLoader, 
              iterative_steps: int, 
              epoch: int, 
              optimizer: Optimizer = None, 
              scheduler: _LRScheduler = None, 
              losses_mapping: dict = None, 
              device: torch.device = 'cpu',
              **kwargs):
        count_ops_and_params = kwargs['count_ops_and_params']
        base_macs, base_nparams = count_ops_and_params(self.model, self.example_input)
        self.logger(f"{ self.model }")
        self.logger(f"base_macs = {float(base_macs) / 1e9:.2f} G, base_nparams = {float(base_nparams) / 1e6:.2f} M")
        self.logger("---------------------------")

        for step in range(iterative_steps):
            
            self.logger(f"[{ step + 1 }/{ iterative_steps }]Start pruning...")

            ## 剪枝
            self.pruner.step()

            ## 微调
            self.logger(f"[{ step + 1 }/{ iterative_steps }]Start fine-tuning...")
            base_train(self.model, train_data=train_data, device=device, epoch=epoch, losses_mapping=losses_mapping, optimizer=optimizer, scheduler=scheduler)

            ## 剪枝后的model
            self.logger(f'{ self.model }')

            ## 计算当此剪枝后的参数
            macs, nparams = count_ops_and_params(self.model, self.example_input)

            self.logger(
               f"  Iter { step + 1 }/{ iterative_steps }, Params: { base_nparams / 1e6:.2f} M => { nparams / 1e6:.2f} M"
            )
            self.logger(
                f"  Iter { step + 1 }/{ iterative_steps }, MACs: { base_macs / 1e9:.2f} G => { macs / 1e9:.2f} G"
            )

            ## 剪枝并微调后进行评测
            acc_1, acc_5 = self.validate(val_data=self.test_data, device=device)
            self.logger(f"after fintune: acc@1: { acc_1 }, acc@5: { acc_5 }")


    def validate(self, val_data: Dataset | DataLoader, criterion: Callable[..., Any] = None, device: torch.device = 'cpu', **kwargs):
        return base_validate(self.model, val_data=val_data, device=device)

    def postPruning(self, **kwargs):
        ## 存储剪枝后的模型
        self.save_model(save_path=kwargs['save_path'], pruning_history = self.pruner.pruning_history(), example_input = self.example_input)

    def save_model(self, save_path: str, **kwargs):
        # base_save_model(self.model, save_path=save_path, tag="model")
        save_dict = { 'model': self.model.state_dict() }
        for key, value in kwargs.items():
            save_dict[key] = value
        point = os.path.join(save_path, "pruned_model.pth")
        torch.save(save_dict, point)

    def get_example_input(self, data: Dataset | DataLoader, device: torch.device, **kwargs):
        if hasattr(self, 'example_input') and self.example_input is not None:
            return self.example_input
        for example, _ in data:
            self.example_input = example[0:1].to(device)
            break
        return self.example_input


class cifarQATEmbarkation(QTAEmbarkationInterface):
    def __init__(self, model: nn.Module, test_data: Dataset | DataLoader, logger: Callable):
        super().__init__(model, test_data, logger)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cpu")
        self.modle = model.to(self.device)


    def preQuantization(self, **kwargs):
        # 量化前的准备，包括 QAT 设置
        self.model.to(self.device)
        self.model = quantModel(self.model)  # 包装模型用于量化
        torch.backends.quantized.engine = 'fbgemm'
        self.model = prepare_for_qat(self.model)  # 准备 QAT 量化
        self.logger("QAT model prepared for training.")

        # 记录模型的参数和 MACs
        self.example_input = self.get_example_input(self.test_data, self.device)
        macs, nparams = count_ops_and_params(self.model, self.example_input)
        print("原模型（量化前）")
        base_macs, base_nparams = count_ops_and_params(self.model, self.example_input)
        self.logger(f"base_macs = {float(base_macs) / 1e9:.2f} G, base_nparams = {float(base_nparams) / 1e6:.2f} M")
        self.logger(f"macs: { macs / 1e9:.2f} G")  # 模型操作数（十亿为单位）
        self.logger("---------------------------")

    def train_with_quantization(self, 
                                train_data: Dataset | DataLoader, 
                                epoch: int, 
                                optimizer: Optimizer = None, 
                                scheduler: _LRScheduler = None, 
                                losses_mapping: dict = None, 
                                device: torch.device = 'cpu',
                                **kwargs):
        best_acc = 0
        for e in range(epoch):
            loop = tqdm(train_data, leave=True)
            self.model.train()
            loss_epoch = 0
            for _, (data, label) in enumerate(loop):
                optimizer.zero_grad()
                data, label = data.to(device), label.to(device)
                output = self.model(data)
                loss = losses_mapping['loss_base']['weight'] * nn.CrossEntropyLoss()(output, label)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                loop.set_description(f'[{ e + 1 } / { epoch }]')
                loop.set_postfix(loss=loss.item())
            loss_epoch /= len(train_data)
            self.logger(f"[{ e + 1 }/{ epoch }] train_loss: { loss_epoch }")
            if scheduler:
                scheduler.step()

            # 验证并保存最优量化模型
            acc_1, acc_5 = self.validate(self.test_data, device=device)
            if acc_1 > best_acc:
                self.save_model(self.logger.save_path)
                best_acc = acc_1
            self.logger(f"[{ e + 1 }/{ epoch }] acc@1: { acc_1 }, acc@5: { acc_5 }, best acc: { best_acc }")
            self.logger("---------------------------")

    def postQuantization(self, **kwargs):
        # 转换为量化模型
        self.model = convert(self.model, inplace=True).to('cpu')
        self.logger("Model converted to quantized format and moved to CPU.")

        # 验证量化后的模型性能
        acc_1, acc_5 = self.validate(self.test_data, device=torch.device('cpu'))
        self.logger(f"Final Quantized Model: acc@1: { acc_1 }, acc@5: { acc_5 }")
        macs, nparams = count_ops_and_params(self.model, self.example_input)
        print(f"nparams: { nparams / 1e6:.2f} M")
        print(f"macs: { macs / 1e9:.2f} G")

    def validate(self, val_data: Dataset | DataLoader, device: torch.device = 'cpu'):
        return base_validate(self.model, val_data=val_data, device=device)

    def save_model(self, save_path: str, **kwargs):
        base_save_model(self.model, save_path=save_path, tag="quantized_model")

    def get_example_input(self, data: Dataset | DataLoader, device: torch.device, **kwargs):
        for example, _ in data:
            return example[0:1].to(device)
        
        

class cifarDynamicQuantizationEmbarkation(DYQEmbarkationInterface):
    def __init__(self, model: nn.Module, test_data: Dataset | DataLoader, logger: Callable):
        super().__init__(model, test_data, logger)
        self.device = torch.device("cpu")  # 动态量化一般在 CPU 上运行
        self.model = model.to(self.device)

    def preQuantization(self, **kwargs):
        # 打印原始模型性能
        self.logger("Preparing model for dynamic quantization...")
        self.example_input = self.get_example_input(self.test_data, self.device)
        macs, nparams = count_ops_and_params(self.model, self.example_input)
        self.logger(f"Original model MACs: {macs / 1e9:.2f} G, Params: {nparams / 1e6:.2f} M")
        acc_1, acc_5 = self.validate(self.test_data, device=self.device)
        self.logger(f"Original Model Performance - acc@1: {acc_1:.4f}, acc@5: {acc_5:.4f}")

    def apply_dynamic_quantization(self):
        # 动态量化模型
        self.logger("Applying dynamic quantization...")
        self.model = quantize_dynamic(self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        self.logger("Dynamic quantization applied successfully.")

    def postQuantization(self, **kwargs):
        # 测试动态量化后的模型性能
        self.logger("Testing dynamically quantized model...")
        acc_1, acc_5 = self.validate(self.test_data, device=self.device)
        self.logger(f"Quantized Model Performance - acc@1: {acc_1:.4f}, acc@5: {acc_5:.4f}")
        macs, nparams = count_ops_and_params(self.model, self.example_input)
        self.logger(f"Quantized model MACs: {macs / 1e9:.2f} G, Params: {nparams / 1e6:.2f} M")

    def validate(self, val_data: Dataset | DataLoader, device: torch.device = 'cpu'):
        return base_validate(self.model, val_data=val_data, device=device)

    def save_model(self, save_path: str, **kwargs):
        base_save_model(self.model, save_path=save_path, tag="dynamic_quantized_model")

    def get_example_input(self, data: Dataset | DataLoader, device: torch.device, **kwargs):
        for example, _ in data:
            return example[0:1].to(device)