import torch
import torch.nn as nn
import torch.nn.functional as F
   
class SDL(nn.Module):
    def __init__(self, num_class=7, dim=768, k=2, size=32):
        super(SDL, self).__init__()
        self.dim = dim
        self.k = k
        self.num_class = num_class
        self.Queue = torch.nn.Parameter(torch.rand(num_class, size, dim),requires_grad=False)
        self.Probe = torch.nn.Parameter(torch.rand(num_class, size, num_class),requires_grad=False)

    def cacu_cosine_similarity(self, Q, x):
        # Q: [num_class, size, dim]
        # x: [dim]
        # 对Q和x进行归一化（转换为单位向量）
        
        x = x.expand(Q.shape[0], Q.shape[1], -1)
        Q_normalized = F.normalize(Q, p=2, dim=2)
        x_normalized = F.normalize(x, p=2, dim=2)
        similarities = F.cosine_similarity(Q_normalized, x_normalized, dim=2)
        # 选出相似度最大的k个
        topk_similarities, topk_indices = similarities.topk(self.k, dim=1)
        return topk_similarities, topk_indices

    def update(self, x, prob, label):
        argmax = label
        # x_prob = F.one_hot(label, self.num_class)
        x_prob = torch.nn.functional.softmax(prob, dim=1)
        # x: [batch_size, dim]
        # argmax: [batch_size]
        for i in range(x.shape[0]):
            queue = torch.cat(
                (self.Queue[argmax[i]], x[i].unsqueeze(0)), dim=0)[1:]
            self.Queue[argmax[i]] = queue
            probe = torch.cat(
                (self.Probe[argmax[i]], x_prob[i].unsqueeze(0)), dim=0)[1:]
            self.Probe[argmax[i]] = probe
            
    @torch.no_grad()
    def forward(self, x, probe, label):
        # x: [batch_size, dim]
        # probe: [batch_size, 7]
        # 更新Queue和Probe
        
        x_probe = probe.detach()
        probe = torch.zeros_like(x_probe, device=probe.device)  # to return
        for i in range(x.shape[0]):
            topk_similarities, topk_indices = self.cacu_cosine_similarity(
                self.Queue, x[i])
            
            # 跟据相似度，更新probe
            p = topk_similarities.unsqueeze(
                2) * self.Probe[torch.arange(self.num_class).unsqueeze(1), topk_indices]
            p = p.reshape(-1, self.num_class)
            probe[i] = torch.sum(p, dim=0) / torch.sum(topk_similarities)
      
        return probe