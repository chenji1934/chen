import torch, torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.m=nn.Sequential(nn.Conv2d(i,o,3,1,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,1,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self,x): return self.m(x)
class BSUVNet2(nn.Module):
    def __init__(self,t=5,in_ch=3,base=32):
        super().__init__(); ic=t*in_ch; self.pool=nn.MaxPool2d(2)
        self.e1,self.e2,self.e3,self.e4=ConvBlock(ic,base),ConvBlock(base,base*2),ConvBlock(base*2,base*4),ConvBlock(base*4,base*8)
        self.bt=ConvBlock(base*8,base*16)
        self.u4,self.d4=nn.ConvTranspose2d(base*16,base*8,2,2),ConvBlock(base*16,base*8)
        self.u3,self.d3=nn.ConvTranspose2d(base*8,base*4,2,2),ConvBlock(base*8,base*4)
        self.u2,self.d2=nn.ConvTranspose2d(base*4,base*2,2,2),ConvBlock(base*4,base*2)
        self.u1,self.d1=nn.ConvTranspose2d(base*2,base,2,2),ConvBlock(base*2,base)
        self.head=nn.Conv2d(base,1,1)
    def forward(self,x):
        b,t,c,h,w=x.shape; x=x.view(b,t*c,h,w)
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); bt=self.bt(self.pool(e4))
        d4=self.d4(torch.cat([self.u4(bt),e4],1)); d3=self.d3(torch.cat([self.u3(d4),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return torch.sigmoid(self.head(d1))
