def compute_metrics(pred,target,threshold=0.5,eps=1e-6):
    p=(pred>=threshold).float(); t=(target>=0.5).float()
    tp=(p*t).sum().item(); fp=(p*(1-t)).sum().item(); fn=((1-p)*t).sum().item(); tn=((1-p)*(1-t)).sum().item()
    precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps); f1=2*precision*recall/(precision+recall+eps); acc=(tp+tn)/(tp+tn+fp+fn+eps); iou=tp/(tp+fp+fn+eps)
    return {"Precision":precision,"Recall":recall,"F1":f1,"Accuracy":acc,"IoU":iou}
def average_dicts(ds): return {k:sum(d[k] for d in ds)/len(ds) for k in ds[0]}
