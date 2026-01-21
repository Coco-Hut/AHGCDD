import copy
from collections import defaultdict
from lib_utils.utils import fix_seed,result_printer
from lib_utils.train_agent import Trainer
from lib_utils.eval_agent import Evaluator
from lib_models.HNN import HCHA,SetGNN
from lib_dataset.data_perturbation import perturbation

class ExpAgent:
    
    def __init__(self,args,**kwargs):
        """
        Overall pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        self.trainer=Trainer(args)
        self.evaluator=Evaluator(args)

    def node_cls_train_eval(self,data):
        
        metrics_dict=defaultdict(list)
        
        src_data = copy.deepcopy(data) 

        for seed in range(self.args.num_seeds):
            
            fix_seed(seed) 
            
            masks=data.generate_random_split(train_ratio=self.args.train_prop,val_ratio=self.args.valid_prop,seed=seed)

            if self.args.is_perturbed:
                data = perturbation(src_data,mode=self.args.pert_mode,p=self.args.pert_p,masks=masks)

            model = parse_model(self.args,data)
            if self.args.method == 'TMPHN':
                pass
            else:
                model = model.to(self.args.device)
                    
            self.trainer.training(model,data,self.args,seed_split=masks,task_type='node_cls')
            
            print(f'------------------------------[Seed {seed}]-----------------------------------')
            result=self.evaluator.evaluate(model,data,seed_split=masks,task_type='node_cls',verbose=True)
            print(f'------------------------------------------------------------------------------')
            
            for m in result:
                metrics_dict[m].append(result[m])
            
        print(f'---------------------------------[Final]--------------------------------------')
        for m in metrics_dict:
            result_printer(metrics_dict[m],m)
        print(f'------------------------------------------------------------------------------')

    def running(self,task_type,data):
        
        if task_type == 'node_cls':
            self.node_cls_train_eval(data)
        else:
            raise NotImplementedError

def parse_model(args, data):
    
    if args.embedding_mode:
        num_targets=args.embedding_hidden
    else:
        num_targets=data.num_classes
    
    # --------- Hypergraph Semi-supervised Models --------------------
    
    if args.method == 'AllSetformer':
        if args.LearnMask:
            #model = SetGNN(data.num_features, num_targets, args)
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method == 'AllDeepSets':
        args.PMA = False 
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(data.num_features, num_targets, args, data.norm)
        else:
            model = SetGNN(data.num_features, num_targets, args)
    elif args.method in ['HGNN','HCHA']:
        model = HCHA(data.num_features, num_targets, args)
    else:
        raise ValueError('Unimplemented model')

    return model