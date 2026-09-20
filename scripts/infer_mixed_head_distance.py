"""Inference on an optical-FOV image; optional fixed full-frame fallback on no head box."""
import argparse,json
import train_mixed_head_distance as exp
from train_mixed_head_distance import torch,cv2,np,ROOT,LABELS,Model,prepare_inputs,image_tensor
@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('image');ap.add_argument('--arm',choices=['B','C'],default='C');ap.add_argument('--checkpoint');ap.add_argument('--fov-diameter',type=float,help='Required for C; optical FOV diameter in input image pixels');ap.add_argument('--fallback-fullframe',action='store_true');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if args.arm=='C' and (args.fov_diameter is None or args.fov_diameter<=0):ap.error('C requires a positive --fov-diameter using the same optical-FOV calibration as training')
    im=cv2.imread(args.image)
    if im is None:raise FileNotFoundError(args.image)
    from ultralytics import YOLO
    detector=YOLO(str(exp.DETECTOR));d=detector.predict(im,imgsz=1024,conf=.25,iou=.7,max_det=100,rect=True,device=args.device,half=False,verbose=False)[0]
    result=dict(image=args.image,arm=args.arm,box=None,detector_confidence=None,prediction='Invalid',probabilities=None,route='no_head')
    if len(d.boxes):
        i=int(d.boxes.conf.argmax());box=d.boxes.xyxy[i].cpu().tolist();result.update(box=box,detector_confidence=float(d.boxes.conf[i]),route='head')
        ck=torch.load(args.checkpoint or exp.OUT/args.arm/'best.pt',map_location='cpu',weights_only=True);assert ck['arm']==args.arm
        model=Model(args.arm,ck['mean'],ck['std']).to(args.device).eval();model.load_state_dict(ck['model_state'])
        _,roi,scale=prepare_inputs(im,box,args.fov_diameter or 1.)
        p=model(image_tensor(roi)[None].to(args.device),torch.tensor(scale)[None].to(args.device)).softmax(-1)[0].cpu().tolist()
        result.update(prediction=LABELS[int(np.argmax(p))],probabilities=dict(zip(LABELS,p)),checkpoint_epoch=ck['epoch'])
    elif args.fallback_fullframe:
        import timm
        ck=torch.load(exp.base.OUT/'best.pt',map_location='cpu',weights_only=True);model=timm.create_model('convnext_tiny',pretrained=False,num_classes=3,drop_rate=.2).to(args.device).eval();model.load_state_dict(ck['model_state'])
        p=model(image_tensor(exp.base.letterbox(im,384))[None].to(args.device)).softmax(-1)[0].cpu().tolist();result.update(route='fullframe_fallback',prediction=LABELS[int(np.argmax(p))],probabilities=dict(zip(LABELS,p)))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
