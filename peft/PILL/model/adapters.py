import torch
import torch.nn as nn
import torch.nn.functional as F

from model.llm_model import ModelArgs
from transformers.activations import get_activation


class NormalAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.input_dim = args.dim
        self.down_sample_size = args.down_sample_size
        self.activation = get_activation(args.adapter_activater.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim, False)

        nn.init.normal_(self.down_sampler.weight, 0, 0.02)
        nn.init.zeros_(self.up_sampler.weight)

    def forward(self, x):
        z = self.down_sampler(x)
        # z = self.activation(z)
        z = self.up_sampler(z)

        return z


class GLUAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.input_dim = args.dim
        self.down_sample_size = args.down_sample_size
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.down_sampler_gate = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim, False)

        nn.init.normal_(self.down_sampler.weight, 0, 0.02)
        nn.init.zeros_(self.up_sampler.weight)

    def forward(self, x):
        z = F.silu(self.down_sampler_gate(x))*self.down_sampler(x)
        # z = self.activation(z)
        z = self.up_sampler(z)
        # z2 = self.up_sampler_gate(z).sigmoid()
        return z
    
adapter_dict = {
    'NormalAdapter': NormalAdapter,
    'GLUAdapter': GLUAdapter
}