import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from lib_utils.utils import mask_to_index
from lib_utils.metrics import evaluate,evaluate_edge,edge_evaluation_printer,hg_evaluation_printer,evaluate_hypegraph
from lib_models import _semi_methods_
from sklearn.utils.class_weight import compute_class_weight
# from lib_models.HNN.preprocessing import algo_preprocessing

class Trainer:
    
    def __init__(self, args, **kwargs):
        """
        Training pipline for different kinds of models
        """
        self.args = args
        self.device=args.device

    def semi_node_cls_training(self,model,data,masks,args):
        
        criterion = nn.NLLLoss()
        
        model.train()
        
        ### Training loop ###
        start_time = time.time()
        model.reset_parameters()
        
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        if args.method == 'HJRL':
            pos_weight_H = float(data.H_ini.shape[0] * data.H_ini.shape[0] - data.H_ini.sum()) / data.H_ini.sum()
            if torch.isinf(pos_weight_H):
                pos_weight_H = torch.FloatTensor(args.pos_weight_thresh)
        elif args.method == 'TMPHN':
            train_idx = mask_to_index(masks['train'].to('cpu'),data.x.shape[0])
            data.target_idx = train_idx

        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            if args.method == 'HJRL':
                node_embed,edge_embed = model(data)
                # 1. node classification loss
                node_embed = F.log_softmax(node_embed, dim=1)
                loss_1 = criterion(node_embed[masks['train']],data.y[masks['train']])
                if args.gamma == 0:
                    loss_2 = 0
                else:
                    # 2. reconstruction loss
                    recovered_H = torch.mm(node_embed, edge_embed.t())
                    recovered_H = torch.sigmoid(recovered_H)
                    if args.sample_ratio:
                        sampled_row = np.random.choice(recovered_H.shape[0], int(recovered_H.shape[0] * args.sample_ratio), replace=False)
                        loss_2 = F.binary_cross_entropy_with_logits(recovered_H[sampled_row], data.H_ini[sampled_row], pos_weight=pos_weight_H)
                    else:
                        loss_2 = F.binary_cross_entropy_with_logits(recovered_H, data.H_ini, pos_weight=pos_weight_H)
                loss = loss_1 + args.gamma * loss_2
                loss.backward()
                optimizer.step()
            else:
                out,_ = model(data) 
                out = F.log_softmax(out, dim=1)
                if args.method == 'TMPHN':
                    loss = criterion(out,data.y[masks['train']])
                else:
                    loss = criterion(out[masks['train']],data.y[masks['train']])
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
                optimizer.step()

            if (epoch+1) % args.display_step == 0:
                
                if args.method == 'TMPHN':
                    data.target_idx=data.all_idx
                    
                result = evaluate(model, data, masks) 
                
                print(f'Epoch: {epoch+1:02d}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test  Acc: {100 * result[2]:.2f}%')

                if args.method == 'TMPHN':
                    data.target_idx=train_idx
                
        end_time = time.time()
        print(f'Training Time: {end_time-start_time:.2f}')
        
        if args.method == 'TMPHN':
            data.target_idx=data.all_idx 
   
    def training(self,model,data,args,seed_split=None,task_type='node_cls'):
        
        if self.args.method in _semi_methods_:
            if task_type == 'node_cls':
                self.semi_node_cls_training(model,data,seed_split,args)
            else:
                raise NotImplementedError