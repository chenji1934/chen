import torch
from torch.utils.data import DataLoader
from config import cfg
from datasets.kt3dmoseg_dataset import KT3DMoSegDataset
from models.bsuvnet2 import BSUVNet2
from utils.metrics import compute_metrics, average_dicts

def main(ckpt='checkpoints/best.pth'):
    device=torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    ds=KT3DMoSegDataset(cfg.DATA_ROOT,'test',(cfg.IMG_HEIGHT,cfg.IMG_WIDTH),cfg.TEMPORAL_WINDOW,cfg.TEMPORAL_STRIDE,cfg.NORMALIZE_MEAN,cfg.NORMALIZE_STD,False)
    ld=DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=False,num_workers=cfg.NUM_WORKERS)
    model=BSUVNet2(t=cfg.TEMPORAL_WINDOW).to(device); model.load_state_dict(torch.load(ckpt,map_location=device)['model']); model.eval()
    mets=[]
    with torch.no_grad():
        for b in ld:
            pred=model(b['clip'].to(device)); mets.append(compute_metrics(pred,b['mask'].to(device),cfg.THRESHOLD))
    m=average_dicts(mets); print(m)
if __name__=='__main__': main()
