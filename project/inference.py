import os, cv2, numpy as np, torch
from config import cfg
from models.bsuvnet2 import BSUVNet2
from utils.io_video import read_video_frames, write_video, overlay_mask

def prep(frame):
    x=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); x=cv2.resize(x,(cfg.IMG_WIDTH,cfg.IMG_HEIGHT)).astype(np.float32)/255.0
    x=(x-np.array(cfg.NORMALIZE_MEAN,dtype=np.float32))/np.array(cfg.NORMALIZE_STD,dtype=np.float32)
    return np.transpose(x,(2,0,1))

def main(video_path='input.mp4', ckpt='checkpoints/best.pth', out_dir='outputs'):
    os.makedirs(out_dir,exist_ok=True); device=torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    model=BSUVNet2(t=cfg.TEMPORAL_WINDOW).to(device); model.load_state_dict(torch.load(ckpt,map_location=device)['model']); model.eval()
    frames=read_video_frames(video_path); T=cfg.TEMPORAL_WINDOW; half=T//2
    masks_v, over_v=[],[]
    with torch.no_grad():
        for i in range(len(frames)):
            idx=[min(max(i+k,0),len(frames)-1) for k in range(-half,half+1)]
            clip=np.stack([prep(frames[j]) for j in idx],0)
            pred=model(torch.from_numpy(clip[None]).float().to(device))[0,0].cpu().numpy()
            mask=(pred>=cfg.THRESHOLD).astype(np.uint8)
            m_raw=cv2.resize(mask*255,(frames[i].shape[1],frames[i].shape[0]),interpolation=cv2.INTER_NEAREST)
            masks_v.append(cv2.cvtColor(m_raw,cv2.COLOR_GRAY2BGR)); over_v.append(overlay_mask(frames[i],(m_raw>0).astype(np.uint8)))
    write_video(masks_v,os.path.join(out_dir,'mask.mp4'),cfg.FPS); write_video(over_v,os.path.join(out_dir,'overlay.mp4'),cfg.FPS)
if __name__=='__main__': main()
