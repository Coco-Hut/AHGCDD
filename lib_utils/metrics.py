import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import f1_score, roc_auc_score,average_precision_score,accuracy_score
from collections import defaultdict
#from lib_models.HNN.preprocessing import algo_preprocessing

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

@torch.no_grad()
def evaluate(model, data, split_idx, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out,_ = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_acc(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_acc(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_acc(
        data.y[split_idx['test']], out[split_idx['test']])
    
    return train_acc, valid_acc, test_acc

def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()

def accuracy(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    accs = []
    for mask in masks.values():
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs

def masked_f1_score(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    f1=f1_score(labels.cpu().numpy(), pred.cpu().numpy()) 
    return f1

def f1_scores(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    f1s = []
    for mask in masks.values():
        f1 = masked_f1_score(logits[mask], labels[mask])
        f1s.append(f1)
    return f1s

def masked_auc_roc(auc_score: Tensor, labels: Tensor):
    if len(auc_score) == 0:
        return 0
    auc_roc=roc_auc_score(labels.cpu().numpy(), auc_score.cpu().numpy())
    return auc_roc

def auc_rocs(logits: Tensor, labels: Tensor, masks: dict[Tensor]):
    probs = F.softmax(logits, dim=1)
    auc_score = probs[:, 1].detach()  
    aucs = []
    for mask in masks.values():
        auc = masked_auc_roc(auc_score[mask], labels[mask])
        aucs.append(auc)
    return aucs

