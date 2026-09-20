"""Single optical-FOV image inference with fixed YOLO11l + four-channel classifier."""
import argparse,json
import train_mixed_rgbmask_distance as x
from train_mixed_rgbmask_distance import torch,cv2,np
@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('image');ap.add_argument('--device',default='cuda:0');ap.add_argument('--checkpoint',default=str(x.OUT/'best.pt'));a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    image=cv2.imread(a.image)
    if image is None:raise FileNotFoundError(a.image)
    from ultralytics import YOLO
    seg=YOLO(str(x.SEG));result=seg.predict(image,imgsz=1024,conf=.8,iou=.7,retina_masks=True,device=a.device,half=False,verbose=False)[0];mask=np.zeros(image.shape[:2],np.uint8)
    if result.masks is not None:
        m=result.masks.data.cpu().numpy();assert m.shape[1:]==mask.shape;mask[(m>.5).any(0)]=255
    inp=torch.cat([x.image_tensor(x.letterbox(image,384)),torch.from_numpy(x.mask_letterbox(mask).astype(np.float32)/255)[None]])[None].to(a.device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=x.make_model(False).to(a.device).eval();model.load_state_dict(ck['model_state']);p=model(inp).softmax(-1)[0].cpu().tolist()
    print(json.dumps(dict(prediction=x.LABELS[int(np.argmax(p))],probabilities=dict(zip(x.LABELS,p)),mask_empty=not bool(mask.any()),epoch=ck['epoch']),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
