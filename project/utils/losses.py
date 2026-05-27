import torch.nn as nn
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-6):
        super().__init__(); self.bce=nn.BCELoss(); self.bw=bce_weight; self.dw=dice_weight; self.eps=eps
    def _dice(self,p,t):
        p=p.view(p.size(0),-1); t=t.view(t.size(0),-1); inter=(p*t).sum(1)
        return 1-((2*inter+self.eps)/(p.sum(1)+t.sum(1)+self.eps)).mean()
    def forward(self,p,t): return self.bw*self.bce(p,t)+self.dw*self._dice(p,t)
