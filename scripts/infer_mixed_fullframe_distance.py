"""Run the mixed full-field model on an image already cropped to its optical FOV."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import torch
import timm
from distance_state_classifier.src.manual_head_abc import letterbox,image_tensor

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('image');ap.add_argument('--checkpoint',default=str(ROOT/'results/convnext_mixed_fullframe_20260919/best.pt'));ap.add_argument('--device',default='cuda:0');ap.add_argument('--output-json');args=ap.parse_args()
    image=cv2.imread(args.image)
    if image is None:raise FileNotFoundError(args.image)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested GPU is unavailable')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=timm.create_model('convnext_tiny',num_classes=3,drop_rate=saved['config']['dropout'],pretrained=False)
    model.load_state_dict(saved['model_state'],strict=True);model.to(args.device).eval()
    prob=model(image_tensor(letterbox(image,saved['input_size']))[None].to(args.device)).softmax(-1)[0].cpu().tolist()
    result={'image':args.image,'prediction':saved['classes'][max(range(len(prob)),key=prob.__getitem__)],'probabilities':dict(zip(saved['classes'],prob)),'checkpoint_epoch':saved['epoch'],'preprocess':saved['preprocess']}
    text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if args.output_json:Path(args.output_json).write_text(text,encoding='utf-8')
if __name__=='__main__':main()
