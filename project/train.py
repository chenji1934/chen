import os, random, numpy as np, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from config import cfg
from datasets.kt3dmoseg_dataset import KT3DMoSegDataset
from models.bsuvnet2 import BSUVNet2
from utils.losses import BCEDiceLoss
from utils.metrics import compute_metrics, average_dicts

def set_seed(s=42): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def build_loader(split, aug=False):
    ds=KT3DMoSegDataset(cfg.DATA_ROOT, split=split, img_size=(cfg.IMG_HEIGHT,cfg.IMG_WIDTH), temporal_window=cfg.TEMPORAL_WINDOW, temporal_stride=cfg.TEMPORAL_STRIDE, mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD, augment=aug)
    return DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=(split=='train'),num_workers=cfg.NUM_WORKERS,pin_memory=True)

def evaluate(model, loader, device):
    model.eval(); mets=[]
    with torch.no_grad():
        for b in loader:
            clip,mask=b['clip'].to(device),b['mask'].to(device)
            pred=model(clip); mets.append(compute_metrics(pred,mask,cfg.THRESHOLD))
    return average_dicts(mets)

def main():
    set_seed(cfg.SEED); os.makedirs(cfg.SAVE_DIR,exist_ok=True); writer=SummaryWriter(cfg.LOG_DIR)
    device=torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader=build_loader('train',True), build_loader('val',False)
    model=BSUVNet2(t=cfg.TEMPORAL_WINDOW).to(device); crit=BCEDiceLoss(); opt=torch.optim.Adam(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=cfg.SCHEDULER_STEP,gamma=cfg.SCHEDULER_GAMMA); scaler=GradScaler(enabled=cfg.AMP)
    best=0.0
    for ep in range(1,cfg.EPOCHS+1):
        model.train(); pbar=tqdm(train_loader,desc='Epoch {}/{}'.format(ep,cfg.EPOCHS)); run=0.0
        for b in pbar:
            clip,mask=b['clip'].to(device),b['mask'].to(device); opt.zero_grad()
            with autocast(enabled=cfg.AMP): pred=model(clip); loss=crit(pred,mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item(); pbar.set_postfix(loss=loss.item())
        sch.step(); tr_loss=run/len(train_loader); val=evaluate(model,val_loader,device)
        writer.add_scalar('Loss/train',tr_loss,ep)
        for k,v in val.items(): writer.add_scalar('Val/{}'.format(k),v,ep)
        if val['F1']>best:
            best=val['F1']; torch.save({'epoch':ep,'model':model.state_dict(),'best_f1':best},os.path.join(cfg.SAVE_DIR,'best.pth'))
        torch.save({'epoch':ep,'model':model.state_dict(),'best_f1':best},os.path.join(cfg.SAVE_DIR,'last.pth'))
        print('Epoch {} | loss {:.4f} | val {}'.format(ep,tr_loss,val))
    writer.close()
if __name__=='__main__': main()
