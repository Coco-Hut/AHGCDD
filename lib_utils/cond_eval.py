# 编写压缩数据train_eval框架
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from collections import defaultdict
from lib_cond import AHGCondX,AHGCond
from lib_utils.utils import fix_seed,result_printer
from lib_utils.metrics import evaluate
from lib_utils.exp_agent import parse_model

class CondData():
    def __init__(self, data):
        super().__init__()
        self.x = data.x
        self.hyperedge_index = data.hyperedge_index
        self.y = data.y
        self.norm = ''

def multi_seed_train_eval(data,args):

    metrics=defaultdict(list) 

    for seed in range(args.num_seeds):
        
        fix_seed(seed) 

        masks=data.generate_random_split(train_ratio=args.train_prop,val_ratio=args.valid_prop,seed=seed)

        if args.cond_method == 'ahgcond':
            agent = AHGCond(data,masks,args)
        elif args.cond_method == 'ahgcondx':
            agent = AHGCondX(data,masks,args)
        else:
            raise NotImplementedError

        cond_data = agent.reduce(data)

        print(f'Node Number for Reduced Hypergraph: {cond_data.x.shape[0]}')
        
        model = parse_model(args,data)
        model = model.to(args.device)

        result = train_eval_cond(model,cond_data,data,masks,args) 
        metrics['acc'].append(np.array(list(result))*100)

    print(f'---------------------------------[Final]--------------------------------------')
    for m in metrics:
        result_printer(metrics[m],m)
    print(f'------------------------------------------------------------------------------')

def train_eval_cond(model,cond_data,data,masks,args):

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

    for epoch in tqdm(range(args.epochs)):
        # Training part
        model.train()
        optimizer.zero_grad()
        out,_ = model(cond_data) 
        out = F.log_softmax(out, dim=1)
        loss = criterion(out,cond_data.y)
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
        optimizer.step()

        if (epoch+1) % args.display_step == 0:
                
            result = evaluate(model, data, masks)
            
            print(f'Epoch: {epoch+1:02d}, '
                f'Train Acc: {100 * result[0]:.2f}%, '
                f'Valid Acc: {100 * result[1]:.2f}%, '
                f'Test  Acc: {100 * result[2]:.2f}%') 

    end_time = time.time()
    print(f'Training Time: {end_time-start_time:.2f}')

    result = evaluate(model, data, masks) 

    return result
