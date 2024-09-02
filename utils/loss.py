import torch

import torch.nn as nn

class KL_div(nn.Module):
    def __init__(self, weight=1.0):
        super(KL_div, self).__init__()
        self.loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.weight = weight
        
    def forward(self, logits, labels):
        soft_logits = torch.nn.functional.log_softmax(logits, dim=1)
        
        loss = self.loss_fn(soft_logits, labels)
        return loss * self.weight
    
    def update_weight(self, epoch):
    #     for i in range(0,60,2):
    # print(i, math.sin(3.14*i/80))
    # print(i,(i/60)**2)
        if epoch < 30:
            self.weight = 0.0
        elif epoch < 50:
            self.weight = 0.01
        elif epoch < 60:
            self.weight = 0.1
        elif epoch < 80:
            self.weight = 0.5
        else:
            self.weight = 1.0
        # self.weight = 0.0    
        # self.weight = (epoch/60)**2
        # self.weight = math.sin(3.14*epoch/60)
        
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.weight = weight
        
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels) * self.weight
    
    def update_weight(self, epoch):
    #     for i in range(0,60,2):
    # print(i, math.sin(3.14*i/80))
    # print(i,(i/60)**2)
        if epoch < 30:
            self.weight = 0.0
        elif epoch < 50:
            self.weight = 0.01
        elif epoch < 60:
            self.weight = 0.1
        elif epoch < 80:
            self.weight = 0.5
        else:
            self.weight = 1.0