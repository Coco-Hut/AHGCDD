from collections import defaultdict
from lib_utils.metrics import *
from lib_models import _semi_methods_
from lib_dataset import _fair_datasets_

class Evaluator:
    
    def __init__(self,args, **kwargs):
        """
        Training pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        
    @torch.no_grad()
    def node_cls_evaluation(self, model, data, masks, result=None):

        if result is not None:
            logits = result
        else:
            model.eval()
            logits,_ = model(data) 

        accs = accuracy(logits, data.y, masks)
        metrics_dict = {'acc': accs}

        return metrics_dict

    def evaluate(self,model,data,seed_split=None,task_type=None,verbose=False):
        
        metrics_dict=defaultdict(list)
        
        if self.args.method in _semi_methods_:
            
            if task_type == 'node_cls':
                
                metrics=self.node_cls_evaluation(model, data, seed_split)
                    
                if verbose:
                    for m in metrics:
                        print(f'train_{m}: {metrics[m][0]:.2f}, valid_{m}: {metrics[m][1]:.2f}, test_{m}: {metrics[m][2]:.2f} ')
                
                metrics_dict = metrics_dict |  metrics     
            
        else:
            raise NotImplementedError
        
        return metrics_dict
    
 