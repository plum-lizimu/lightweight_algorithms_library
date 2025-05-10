'''
    Model evaluation function
'''
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

def val_model(model, dataloader, device):
    model.eval()
    y_pre = []
    y_true = []
    val_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    features = {
        'user_feature': None,
        'item_feature': None
    }
    for data in tqdm(dataloader):
        features['user_feature'], features['item_feature'], labels = data['user'].to(device), data['item'].to(device), data['label']
        with torch.no_grad():
            output = model(features)
            loss = loss_fn(output, labels.to(device))
            logits = torch.sigmoid(output)
            y_pre.append(logits.detach().cpu().numpy())
            y_true.append(labels.numpy())
            val_loss += loss.item()
            # loop.set_postfix(loss = loss.item())
    y_pre = np.concatenate(y_pre)
    y_true = np.concatenate(y_true)
    val_loss /= len(dataloader)
    auc = roc_auc_score(y_true, y_pre)
    logLoss = log_loss(y_true, y_pre)
    return val_loss, auc, logLoss