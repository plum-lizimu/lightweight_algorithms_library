import torch
import torch.nn as nn
import torch.nn.functional as F

class NormKD(nn.Module):
	'''
	NormKD: Normalized Logits for Knowledge Distillation
	https://arxiv.org/pdf/2308.00520
	'''
	def __init__(self, t_norm):
		super().__init__()
		self.t_norm = t_norm

	def forward(self, logits_student, logits_teacher):
		
		tstd=logits_teacher.std(dim=1,keepdim=True)
		sstd=logits_student.std(dim=1,keepdim=True)

		dywt=tstd * self.t_norm
		dyws=sstd * self.t_norm

		rt=(logits_teacher)/dywt
		rs=(logits_student)/dyws

		log_pred_student = F.log_softmax(rs, dim=1)
		pred_teacher = F.softmax(rt, dim=1)
		
		loss_kd = (F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1,keepdim=True)*(dywt**2)).mean()

		return loss_kd

