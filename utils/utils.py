import copy

import torch
import torch.nn as nn

from torchmetrics import AUROC, Accuracy
from torchsurv.metrics.cindex import ConcordanceIndex
from collections import ChainMap


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def load_pretrained_model(model, path, model_name='transformer'):
    print(f'Loading pretrained {model_name}...')
    state_dict = torch.load(path)['state_dict']

    pretrained_dict = {}
    for name in state_dict.keys():
        if name.startswith(model_name):
            new_name = '.'.join(name.split('.')[1:])
            pretrained_dict[new_name] = state_dict[name]
    
    model.load_state_dict(pretrained_dict)
    return model

class CIndexWrapper:
    """Wrapper for ConcordanceIndex to provide an update method compatible with trainer.py"""
    def __init__(self):
        self.cindex = ConcordanceIndex()
        self.preds = []
        self.events = []
        self.times = []
        
    def update(self, preds, events, times):
        # Store predictions, events and times for later computation
        self.preds.append(preds.detach().cpu())
        self.events.append(events.detach().cpu())
        self.times.append(times.detach().cpu())
        
    def compute(self):
        # Concatenate all batches and compute concordance index
        if not self.preds:
            return 0.0
            
        all_preds = torch.cat(self.preds, dim=0)
        all_events = torch.cat(self.events, dim=0)
        all_times = torch.cat(self.times, dim=0)
        
        return self.cindex(all_preds, all_events, all_times)
        
    def reset(self):
        # Clear stored data
        self.preds = []
        self.events = []
        self.times = []

class Metric:
    "Metrics dispatcher. Adapted from answer at https://stackoverflow.com/a/58923974"
    def __init__(self, num_classes):
        self.num_classes=num_classes

    def get_metric(self, metric='acc'):
        """Dispatch metric with method"""
        
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, metric, lambda: "Metric not implemented yet")
        
        return method()

    def auroc(self):
        return AUROC(num_classes=self.num_classes, task="binary")

    def acc(self):
        return  Accuracy(num_classes=self.num_classes)
    
    def cindex(self):
        return CIndexWrapper()



    


    

