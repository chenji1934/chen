import cv2, numpy as np
def read_video_frames(video_path):
    cap=cv2.VideoCapture(video_path); frames=[]
    if not cap.isOpened(): raise RuntimeError('无法打开视频: {}'.format(video_path))
    while True:
        ret,f=cap.read()
        if not ret: break
        frames.append(f)
    cap.release(); return frames
def write_video(frames,out_path,fps=25):
    h,w=frames[0].shape[:2]; vw=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for f in frames: vw.write(f)
    vw.release()
def overlay_mask(frame_bgr,mask):
    c=np.zeros_like(frame_bgr); c[:,:,2]=(mask*255).astype(np.uint8)
    return cv2.addWeighted(frame_bgr,0.75,c,0.25,0)
