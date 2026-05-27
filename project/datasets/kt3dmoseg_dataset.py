import os, random, cv2, numpy as np, torch
from glob import glob
from torch.utils.data import Dataset
class SpatioTemporalAugment:
    def __call__(self, frames, mask):
        if random.random()<0.5: frames=[cv2.flip(f,1) for f in frames]; mask=cv2.flip(mask,1)
        if random.random()<0.3:
            a,b=random.uniform(0.8,1.2),random.uniform(-20,20)
            frames=[cv2.convertScaleAbs(f,alpha=a,beta=b) for f in frames]
        return frames, mask
class KT3DMoSegDataset(Dataset):
    def __init__(self, root, split="train", img_size=(256,256), temporal_window=5, temporal_stride=1, mean=None, std=None, augment=False):
        self.root=root; self.split=split; self.h,self.w=img_size; self.tw=temporal_window; self.ts=temporal_stride
        self.mean=np.array(mean or [0.485,0.456,0.406],dtype=np.float32); self.std=np.array(std or [0.229,0.224,0.225],dtype=np.float32)
        self.aug=SpatioTemporalAugment() if augment else None; self.samples=self._build_index()
    def _build_index(self):
        samples=[]; split_root=os.path.join(self.root,self.split)
        for scene in sorted([d for d in glob(os.path.join(split_root,'*')) if os.path.isdir(d)]):
            inp,gt=os.path.join(scene,'input'),os.path.join(scene,'groundtruth')
            fs,gs=sorted(glob(os.path.join(inp,'*.*'))),sorted(glob(os.path.join(gt,'*.*')))
            gm={os.path.basename(p):p for p in gs}; half=(self.tw//2)*self.ts
            for i in range(half,len(fs)-half):
                idx=[i+k*self.ts for k in range(-(self.tw//2),self.tw//2+1)]
                clip=[fs[j] for j in idx]; name=os.path.basename(fs[i])
                if name in gm: samples.append((clip,gm[name]))
        if not samples: raise RuntimeError('未找到样本，请检查目录结构')
        return samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        cps,gp=self.samples[idx]
        frames=[cv2.cvtColor(cv2.resize(cv2.imread(p), (self.w,self.h)), cv2.COLOR_BGR2RGB) for p in cps]
        mask=(cv2.resize(cv2.imread(gp,0),(self.w,self.h),interpolation=cv2.INTER_NEAREST)>127).astype(np.float32)
        if self.aug: frames,mask=self.aug(frames,mask)
        arr=[]
        for f in frames:
            f=f.astype(np.float32)/255.0; f=(f-self.mean)/self.std; arr.append(np.transpose(f,(2,0,1)))
        return {"clip":torch.from_numpy(np.stack(arr,0)).float(),"mask":torch.from_numpy(mask[None]).float()}
