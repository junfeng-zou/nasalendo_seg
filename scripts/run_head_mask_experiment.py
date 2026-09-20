"""Controlled soft-mask input experiment: same detector and ConvNeXt architectures."""
import argparse, copy, csv, hashlib, html, json, random, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from torch import nn
from distance_state_classifier.src.manual_head_abc import FusionHead,make_backbone,prepare_inputs,image_tensor
from evaluate_forceps_head_replacement import metrics,LABELS
OLD=ROOT/'results/forceps_head_yolo11s_20260917'
CLS=ROOT/'results/convnext_manual_head_abc_20260917'
OUT=ROOT/'results/head_mask_experiment_20260918'
SEG=ROOT/'larger_surgical_video_dataset_training_20260712/training_results/yolo11/yolo11l_updated_20260712_223026/weights/best.pt'
ARMS=['B_roi','C_roi_scale']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text())
def soft_mask(image,mask,background=.5):
    assert image.shape[:2]==mask.shape and image.dtype==np.uint8
    weight=background+(1-background)*(mask.astype(np.float32)/255)
    return np.rint(image.astype(np.float32)*weight[:,:,None]).clip(0,255).astype(np.uint8)
def setup():
    assert torch.cuda.is_available(),'GPU required'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def prepare():
    from ultralytics import YOLO
    if (OUT/'prepared.json').exists():raise FileExistsError('Prepared output exists')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['masks','preview','classifier']: (OUT/name).mkdir(exist_ok=True)
    audit=read(OLD/'data_audit.json');model=YOLO(str(SEG));records=[];cards=[]
    frozen=[SEG,OLD/'initial_yolo11s.pt',CLS/'frozen_backbone.pt',OLD/'comparison_predictions.json']+[CLS/f'runs/{a}/seed_{s}/best.pt' for a in ARMS for s in [42,43,44]]
    hashes={str(p):sha(p) for p in frozen}
    for index,r in enumerate(audit['records']):
        assert sha(r['source'])==r['image_sha256'] and sha(r['annotation'])==r['annotation_sha256']
        image=cv2.imread(r['source'])
        # Native-size masks avoid resizing a letterboxed mask without removing padding.
        result=model.predict(image,imgsz=1024,conf=.8,iou=.7,retina_masks=True,device=0,half=False,verbose=False)[0]
        mask=np.zeros(image.shape[:2],np.uint8)
        if result.masks is not None:
            masks=result.masks.data.cpu().numpy();assert masks.shape[1:]==mask.shape
            mask[(masks>.5).any(0)]=255
        masked=soft_mask(image,mask)
        dst=OUT/'dataset/images'/r['split']/(r['sample_id']+'.png');dst.parent.mkdir(parents=True,exist_ok=True)
        label=OUT/'dataset/labels'/r['split']/(r['sample_id']+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OLD/'dataset/labels'/r['split']/label.name,label)
        maskpath=OUT/'masks'/(r['sample_id']+'.png')
        assert cv2.imwrite(str(maskpath),mask) and cv2.imwrite(str(dst),masked)
        record={**r,'masked_image':str(dst),'mask_path':str(maskpath),'mask_sha256':sha(maskpath),'masked_sha256':sha(dst),'mask_area_fraction':float((mask>0).mean()),'mask_empty':not bool(mask.any())}
        if r['box'] is not None:
            x1,y1,x2,y2=map(int,r['box']);record['head_box_mask_fraction']=float((mask[y1:y2,x1:x2]>0).mean())
        records.append(record)
        cv2.imwrite(str(OUT/'preview'/(r['sample_id']+'.jpg')),cv2.resize(np.concatenate([image,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),masked],axis=1),(1242,360)))
        cards.append(f'<article><h3>{r["sample_id"]} / {r["split"]}</h3><p>mask像素占比={record["mask_area_fraction"]:.3f}</p><img width="1242" src="preview/{r["sample_id"]}.jpg"></article>')
        if (index+1)%30==0:print('[masks]',index+1,'/180',flush=True)
    (OUT/'dataset.yaml').write_text(f'path: {OUT/"dataset"}\ntrain: images/train\nval: images/val\nnames:\n  0: forceps_head\n')
    config=read(OLD/'train_config.json');config.update(data=str(OUT/'dataset.yaml'),project=str(OUT),name='detector_train')
    dump(OUT/'detector_config.json',config)
    dump(OUT/'prepared.json',dict(records=records,frozen_hashes=hashes,mask_model=str(SEG),segmentation=dict(imgsz=1024,conf=.8,iou=.7,retina_masks=True,selection='union of all detected instrument instances; no head GT used'),input_rule='RGB * (0.5 + 0.5 * predicted_binary_instrument_mask)',background=.5))
    (OUT/'mask_preview.html').write_text('<!doctype html><meta charset="utf-8"><h1>原图 / 预测器械掩码 / 软背景抑制输入</h1>'+''.join(cards))

def verify():
    data=read(OUT/'prepared.json')
    for path,digest in data['frozen_hashes'].items():assert sha(path)==digest,path
    for r in data['records']:
        for path,key in [('source','image_sha256'),('annotation','annotation_sha256'),('mask_path','mask_sha256'),('masked_image','masked_sha256')]:assert sha(r[path])==r[key]
    return data

def train_detector():
    from ultralytics import YOLO
    verify()
    if (OUT/'detector_train').exists():raise FileExistsError('Detector run already exists')
    YOLO(str(OLD/'initial_yolo11s.pt')).train(**read(OUT/'detector_config.json'))

def train_classifiers():
    data=verify();out=OUT/'classifier'
    if (out/'runs').exists():raise FileExistsError('Classifier runs already exist')
    config=read(CLS/'config.json');config['output']=str(out);config['input_variant']='predicted mask soft background suppression 0.5';dump(out/'config.json',config)
    prepared=read(CLS/'prepared.json');by_id={r['sample_id']:r for r in data['records']}
    records=prepared['records'];backbone=make_backbone().cuda().eval()
    backbone.load_state_dict(torch.load(CLS/'frozen_backbone.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    features=[]
    for start in range(0,len(records),8):
        inputs=[]
        for r in records[start:start+8]:
            assert r['box']==by_id[r['sample_id']]['box']
            im=cv2.imread(by_id[r['sample_id']]['masked_image'])
            _,roi,_=prepare_inputs(im,r['box'],r['fov_diameter'])
            inputs.append(image_tensor(roi))
        with torch.inference_mode():features.append(backbone(torch.stack(inputs).cuda()).cpu())
    visual=torch.cat(features).cuda();torch.save(dict(sample_ids=[r['sample_id'] for r in records],roi=visual.cpu()),out/'frozen_features.pt')
    train_indices=torch.tensor([i for i,r in enumerate(records) if r['split']=='train'],device='cuda')
    val_indices=torch.tensor([i for i,r in enumerate(records) if r['split']=='val'],device='cuda')
    target=torch.tensor([r['target'] for r in records],device='cuda')
    raw=torch.tensor([r['scale_raw'] for r in records],device='cuda')
    mean=raw[train_indices].mean(0);std=raw[train_indices].std(0,unbiased=False).clamp_min(1e-6)
    # Use exact original float32 numpy-derived normalizer, not a new numerical variant.
    mean=torch.tensor(prepared['audit']['scale_mean_train'],device='cuda');std=torch.tensor(prepared['audit']['scale_std_train'],device='cuda')
    predictions=[];results=[]
    def score(indices,logits):
        preds=logits.argmax(-1).cpu().tolist()
        return metrics([{'label':LABELS[int(target[i])],'prediction':LABELS[p]} for i,p in zip(indices,preds)],'prediction')
    for seed in [42,43,44]:
        for arm in ARMS:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            head=FusionHead(config['hidden'],config['dropout']).cuda()
            initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
            old=torch.load(CLS/f'runs/{arm}/seed_{seed}/best.pt',map_location='cpu',weights_only=True)
            assert initial_hash==old['initial_head_sha256']
            scales=(raw-mean)/std if arm=='C_roi_scale' else torch.zeros_like(raw)
            optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
            generator=torch.Generator().manual_seed(seed);best=-1;stale=0;history=[]
            run=out/f'runs/{arm}/seed_{seed}';run.mkdir(parents=True)
            for epoch in range(1,config['epochs']+1):
                head.train();order=torch.randperm(len(train_indices),generator=generator).cuda()
                for start in range(0,len(order),config['batch_size']):
                    idx=train_indices[order[start:start+config['batch_size']]];optimizer.zero_grad(set_to_none=True)
                    loss=nn.functional.cross_entropy(head(visual[idx],scales[idx]),target[idx]);assert torch.isfinite(loss)
                    loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
                head.eval()
                with torch.inference_mode():val=score(val_indices,head(visual[val_indices],scales[val_indices]))
                history.append(dict(epoch=epoch,val=val))
                if val['macro_f1']>best+1e-12:
                    best=val['macro_f1'];stale=0
                    payload=dict(model_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},config=config,arm=arm,seed=seed,epoch=epoch,scale_mean=mean.cpu(),scale_std=std.cpu(),validation=val,initial_head_sha256=initial_hash)
                    torch.save(payload,run/'best.pt')
                else:stale+=1
                if stale>=config['patience']:break
            head.load_state_dict(payload['model_state']);head.eval()
            with torch.inference_mode():probs=head(visual,scales).softmax(-1).cpu().numpy()
            for r,p in zip(records,probs):predictions.append(dict(arm=arm,seed=seed,sample_id=r['sample_id'],split=r['split'],label=r['label'],prediction=LABELS[int(p.argmax())],probs=p.tolist()))
            result=dict(arm=arm,seed=seed,best_epoch=payload['epoch'],last_epoch=epoch,val=payload['validation']);results.append(result);dump(run/'history.json',history)
            print('[classifier]',result,flush=True)
    dump(out/'predictions.json',predictions);dump(out/'results.json',results);verify()
    dump(out/'completion.json',dict(runs=6,backbone=str(CLS/'frozen_backbone.pt'),backbone_sha256=sha(CLS/'frozen_backbone.pt')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','detector','classifier']);args=parser.parse_args();setup()
    {'prepare':prepare,'detector':train_detector,'classifier':train_classifiers}[args.stage]()
