"""Export completed X-AnyLabeling annotations and train a one-class detector."""
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from distance_state_classifier.src.manual_head_abc import parse_head_box
OUT = ROOT / 'results/forceps_head_yolo11s_20260917'
PKG = ROOT / 'annotation_projects/forceps_head_pilot_20260907'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[]
    for r in csv.DictReader((PKG/'manifest.csv').open(encoding='utf-8-sig')):
        split=r['annotation_split']; assert split in ('train','val')
        src=PKG/r['annotation_image']; ann=PKG/r['expected_annotation_json']
        im=cv2.imread(str(src)); assert im is not None
        h,w=im.shape[:2]; box=parse_head_box(json.loads(ann.read_text()),w,h)
        dst=OUT/'dataset/images'/split/src.name
        lab=OUT/'dataset/labels'/split/(src.stem+'.txt')
        dst.parent.mkdir(parents=True,exist_ok=True); lab.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        text=''
        if box:
            x1,y1,x2,y2=box
            text=f'0 {(x1+x2)/2/w:.9f} {(y1+y2)/2/h:.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}\n'
        lab.write_text(text)
        records.append(dict(sample_id=r['sample_id'],split=split,image=str(dst),source=str(src),annotation=str(ann),box=box,label=r['distance_label'],video_id=r['video_id'],image_sha256=sha(src),annotation_sha256=sha(ann)))
    assert len(records)==180
    assert not ({r['image_sha256'] for r in records if r['split']=='train'} & {r['image_sha256'] for r in records if r['split']=='val'})
    (OUT/'dataset.yaml').write_text(f'path: {OUT / "dataset"}\ntrain: images/train\nval: images/val\nnames:\n  0: forceps_head\n')
    init=OUT/'initial_yolo11s.pt'
    if not init.exists(): shutil.copy2(ROOT/'weights/yolo11s.pt',init)
    c=ROOT/'results/convnext_manual_head_abc_20260917'
    config=dict(data=str(OUT/'dataset.yaml'),epochs=100,patience=25,imgsz=1024,batch=8,device=0,workers=0,seed=42,deterministic=True,amp=False,optimizer='AdamW',lr0=.001,lrf=.01,weight_decay=.0005,warmup_epochs=3,mosaic=0.0,mixup=0.0,copy_paste=0.0,degrees=10.0,translate=.05,scale=.2,fliplr=.5,flipud=0.0,hsv_h=.015,hsv_s=.3,hsv_v=.3,project=str(OUT),name='train',exist_ok=False,plots=True,save=True)
    dump(OUT/'data_audit.json',dict(records=records,counts={s:dict(total=sum(r['split']==s for r in records),positive=sum(r['split']==s and r['box'] is not None for r in records)) for s in ('train','val')},initial_weights_sha256=sha(init),frozen_classifier_hashes={str(p):sha(p) for p in (c/'frozen_backbone.pt',c/'runs/C_roi_scale/seed_42/best.pt')},protocol=dict(confidence=.25,iou_nms=.7,selection='highest confidence; no ground-truth based selection',classifier_seed=42,empty_labels='Completed empty annotations are negatives for visible-head detection; 42 explicitly confirmed, 10 earlier empty labels retained.',limitation='Mixed-frame same-video development validation, not independent clinical evaluation.')))
    dump(OUT/'train_config.json',config)
    return config
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if not args.prepare_only and (OUT/'train').exists():
        raise RuntimeError('Training output exists; preserve this run and use a new output directory for another experiment')
    cfg=prepare()
    if not args.prepare_only:
        import torch
        from ultralytics import YOLO
        assert torch.cuda.is_available(), 'GPU required'
        torch.set_num_threads(4)
        YOLO(str(OUT/'initial_yolo11s.pt')).train(**cfg)
