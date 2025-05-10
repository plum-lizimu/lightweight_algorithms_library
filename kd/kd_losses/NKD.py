import torch
import torch.nn as nn
import torch.nn.functional as F

class NKD(nn.Module):
	'''
	From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach with Normalized Loss and Customized Soft Labels
	https://arxiv.org/pdf/2303.13005
	'''
	def __init__(self, gamma, temperature=1.):
		super().__init__()
		self.gamma = gamma
		self.temperature = temperature

	def forward(self, logits_student, logits_teacher, label):
		
		target = label.reshape(len(label), -1)

		N, c = logits_student.shape
		# 函数中的第一部分
		log_pred_student = F.log_softmax(logits_student, dim=1)
		pred_teacher = F.softmax(logits_teacher, dim=1)

		target_student = torch.gather(log_pred_student, 1, target)  # gather的作用是把目标类别的值取出来
		target_teacher = torch.gather(pred_teacher, 1, target)  # shape: (batch_size,1)
		tckd_loss = -(target_student * target_teacher).mean()

		# 函数中的第二部分
		mask = torch.ones_like(logits_student).scatter_(1, target, 0).bool()  # scatter的作用是把目标类别的mask置0
		logits_student = logits_student[mask].reshape(N, -1)
		logits_teacher = logits_teacher[mask].reshape(N, -1)

		non_target_student = F.log_softmax(logits_student / self.temperature, dim=1)
		non_target_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

		nckd_loss = -(non_target_student * non_target_teacher).sum(dim=1).mean()
		return tckd_loss + self.gamma * (self.temperature ** 2) * nckd_loss

