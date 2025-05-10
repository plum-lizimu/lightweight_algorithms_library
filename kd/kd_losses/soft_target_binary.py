
import torch.nn as nn
import torch.nn.functional as F

class soft_target_binary(nn.Module):
	
	def __init__(self, T=1.):
		super().__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.binary_cross_entropy_with_logits(out_s / self.T, 
								                    F.sigmoid(out_t / self.T))

		return loss
