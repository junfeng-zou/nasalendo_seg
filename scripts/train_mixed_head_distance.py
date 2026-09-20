"""Fixed automatic head detector + mixed-domain B/C fine-tuning and paired evaluation."""
import sys,json,time,random,argparse,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_mixed_fullframe_distance as base
from train_mixed_fullframe_distance import ROOT,LABELS,sha,dump,read,torch,np,cv2,nn,DataLoader,Counter
from distance_state_classifier.src.manual_head_abc import prepare_inputs,make_backbone,FusionHead,image_tensor
DATA=ROOT/'datasets/mixed_head_distance_20260919'
OUT=ROOT/'results/convnext_mixed_head_20260919'
DETECTOR=ROOT/'results/forceps_head_yolo11s_20260917/train/weights/best.pt'
SOURCE=base.DATA/'manifest.json'
def prepare():
    assert not (DATA/'manifest.json').exists()
    (DATA/'inputs').mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    detector=YOLO(str(DETECTOR));clinical={r['sample_id']:r for r in map(json.loads,(ROOT/'results/clinical_keyframes_self_zjf_20260919/frame_predictions.jsonl').read_text().splitlines())}
    calibration=read(ROOT/'results/convnext_head_roi_cache_20260907/audit.json')['calibration']
    seen={r['image_sha256'] for r in read(ROOT/'results/forceps_head_yolo11s_20260917/data_audit.json')['records'] if r['split']=='train'}
    rows=[]
    for i,original in enumerate(read(SOURCE)['records']):
        r=dict(original);assert sha(r['image_path'])==r['image_sha256'];im=cv2.imread(r['image_path'])
        if r['domain']=='clinical':
            old=clinical[r['sample_id'].removeprefix('clinical__')];assert old['image_sha256']==r['image_sha256']
            box=old['box'];confidence=old['detector_confidence'];diameter=old['fov_diameter']
        else:
            prediction=detector.predict(im,imgsz=1024,conf=.25,iou=.7,max_det=100,rect=True,device=0,half=False,verbose=False)[0]
            boxes=prediction.boxes;box=None;confidence=None
            if len(boxes):
                index=int(boxes.conf.argmax());box=boxes.xyxy[index].cpu().tolist();confidence=float(boxes.conf[index])
            diameter=calibration[r['video_id']]['diameter']
        r.update(box=box,detector_confidence=confidence,fov_diameter=diameter,detector_training_overlap=r['image_sha256'] in seen,roi_path=None,scales=None)
        if box is not None:
            _,roi,scales=prepare_inputs(im,box,diameter);path=DATA/'inputs'/(r['sample_id']+'.png');assert cv2.imwrite(str(path),roi)
            r.update(roi_path=str(path),roi_sha256=sha(path),scales=scales.tolist())
        rows.append(r)
        if (i+1)%200==0:print('prepare',i+1,flush=True)
    dump(DATA/'manifest.json',dict(records=rows,source_manifest_sha256=sha(SOURCE),detector_sha256=sha(DETECTOR),clinical_cache_sha256=sha(ROOT/'results/clinical_keyframes_self_zjf_20260919/frame_predictions.jsonl')))
    print('coverage',dict(Counter((r['domain'],r['split'],r['box'] is not None) for r in rows)),flush=True)
class Dataset(base.CachedImages):
    def __init__(self,rows,augmentation=None):
        super().__init__([dict(r,input_path=r['roi_path']) for r in rows],augmentation)
    def __getitem__(self,i):
        image,target,index=super().__getitem__(i)
        return image,torch.tensor(self.rows[i]['scales']),target
class Model(nn.Module):
    def __init__(self,arm,mean,std,pretrained=False):
        super().__init__();self.backbone=make_backbone(base.PRETRAIN if pretrained else None);self.head=FusionHead();self.arm=arm
        self.register_buffer('scale_mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale_std',torch.tensor(std,dtype=torch.float32))
    def forward(self,image,scale):
        aux=(scale-self.scale_mean)/self.scale_std if self.arm=='C' else torch.zeros_like(scale)
        return self.head(self.backbone(image),aux)
def predict(model,rows):
    model.eval();probs=[]
    with torch.inference_mode():
        for image,scale,_ in DataLoader(Dataset(rows),batch_size=16):
            probs.extend(model(image.cuda(),scale.cuda()).softmax(-1).cpu().tolist())
    return probs
def metric(rows,preds):
    cm=np.zeros((3,4),dtype=int)
    for r,p in zip(rows,preds):cm[LABELS.index(r['label']),LABELS.index(p) if p in LABELS else 3]+=1
    tp=np.diag(cm[:,:3]);support=cm.sum(1);precision=tp/np.maximum(cm[:,:3].sum(0),1);recall=tp/np.maximum(support,1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(n=len(rows),correct=int(tp.sum()),accuracy=float(tp.sum()/len(rows)),macro_f1=float(f1.mean()),recall=recall.tolist(),confusion_matrix=cm.tolist())
def train(arm):
    dest=OUT/arm;dest.mkdir(exist_ok=True);assert not (dest/'best.pt').exists()
    manifest=read(DATA/'manifest.json');assert manifest['source_manifest_sha256']==sha(SOURCE);assert manifest['detector_sha256']==sha(DETECTOR)
    rows=manifest['records'];trainrows=[r for r in rows if r['split']=='train' and r['box'] is not None]
    for r in rows:
        if r['box'] is not None:assert sha(r['roi_path'])==r['roi_sha256']
    cfg=read(base.OUT/'config.json');cfg.update(preprocess='Highest-confidence YOLO head box -> 1.2x expansion -> letterbox384 RGB -> ImageNet normalization; B zero auxiliary; C train-standardized original box width/FOV and height/FOV',arm=arm,architecture='ConvNeXt-Tiny + LayerNorm768 + Linear770,128/GELU/Dropout0.2/Linear128,3; B has two zero auxiliaries',selection='mean domain validation macro F1, no-box predictions Invalid; FP32',manifest_sha256=sha(DATA/'manifest.json'))
    dump(dest/'config.json',cfg)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    scales=np.array([r['scales'] for r in trainrows]);mean=scales.mean(0);std=np.maximum(scales.std(0),1e-6)
    model=Model(arm,mean,std,True).cuda()
    loader=DataLoader(Dataset(trainrows,cfg['augmentation']),batch_size=16,shuffle=True,num_workers=2,pin_memory=True,worker_init_fn=base.seed_worker,generator=torch.Generator().manual_seed(42))
    optimizer=torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=1e-5),dict(params=model.head.parameters(),lr=3e-4)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,40);scaler=torch.cuda.amp.GradScaler()
    counts=np.bincount([LABELS.index(r['label']) for r in trainrows],minlength=3);weights=1/counts;weights/=weights.mean();criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'))
    history=[];best=-1;stale=0;start=time.time()
    for epoch in range(1,41):
        model.backbone.requires_grad_(epoch>3);model.train();losses=[]
        for image,scale,target in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():loss=criterion(model(image.cuda(),scale.cuda()),target.cuda())
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update();losses.append(loss.item())
        scores={}
        for domain in ['phantom','clinical']:
            subset=[r for r in rows if r['split']=='val' and r['domain']==domain];available=[r for r in subset if r['box'] is not None]
            outputs={r['sample_id']:LABELS[int(np.argmax(p))] for r,p in zip(available,predict(model,available))}
            scores[domain]=metric(subset,[outputs.get(r['sample_id'],'Invalid') for r in subset])
        score=np.mean([v['macro_f1'] for v in scores.values()]);scheduler.step();history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=scores,score=float(score)))
        if score>best+1e-12:
            best=score;stale=0;torch.save(dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},arm=arm,mean=mean.tolist(),std=std.tolist(),epoch=epoch,config=cfg),dest/'best.pt')
        else:stale+=1
        dump(dest/'history.json',history);print(arm,epoch,'loss',round(np.mean(losses),4),'score',round(score,4),'stale',stale,flush=True)
        if epoch>3 and stale>=10:break
    dump(dest/'completion.json',dict(seconds=time.time()-start,last_epoch=epoch,checkpoint_sha256=sha(dest/'best.pt')))
def evaluate():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=read(DATA/'manifest.json')['records'];baseline={r['sample_id']:r['prediction'] for r in csv.DictReader((base.OUT/'predictions.csv').open(encoding='utf-8-sig'))};evaluated=[r for r in rows if r['split']!='train'];predictions={};results={}
    for arm in ['B','C']:
        ck=torch.load(OUT/arm/'best.pt',map_location='cpu',weights_only=True);model=Model(arm,ck['mean'],ck['std']).cuda();model.load_state_dict(ck['model_state']);available=[r for r in evaluated if r['box'] is not None]
        predictions[arm]={r['sample_id']:LABELS[int(np.argmax(p))] for r,p in zip(available,predict(model,available))}
    detail=[]
    for r in evaluated:
        s=r['sample_id'];detail.append(dict(**r,fullframe=baseline[s],B=predictions['B'].get(s,'Invalid'),C=predictions['C'].get(s,'Invalid'),B_fallback=predictions['B'].get(s,baseline[s]),C_fallback=predictions['C'].get(s,baseline[s])))
    for domain,split in [('clinical','val'),('phantom','val'),('phantom','test')]:
        subset=[r for r in detail if r['domain']==domain and r['split']==split];detected=[r for r in subset if r['box'] is not None]
        results[domain+'_'+split]={key:metric(subset,[r[key] for r in subset]) for key in ['fullframe','B','C','B_fallback','C_fallback']}
        results[domain+'_'+split]['detected_only']={key:metric(detected,[r[key] for r in detected]) for key in ['fullframe','B','C']}
    dump(OUT/'results.json',results);dump(OUT/'predictions.json',detail)
    lines=['# 自动头部框＋混合数据重训 B/C','', '固定原 YOLO11s 头部检测器；仅重新训练距离分类器。不使用人工框或器械掩码。B=RGB头部裁剪；C=同样裁剪＋原框宽高/光学视野直径。框扩大1.2倍后等比例填充384。', '', '沿用混合整图实验的病例划分、ImageNet初始化、40轮上限、3轮冻结、学习率、增强和种子42。B/C使用相同FusionHead128；与整图原生线性头不同。主干均微调。只用训练集有框样本拟合尺度标准化和类别权重；无框训练样本明确排除。', '', '按两个域完整验证集的Macro-F1均值选权重；无框视为Invalid并计入漏分。最终FP32。测试集不参与选权重。回退方案在无框时使用既有固定整图分类器，不能视作纯头部模型性能。', '', '## 数据覆盖','', '|域/划分|全部|有框|检测器训练见过|','|---|---:|---:|---:|']
    for domain,split in [('phantom','train'),('clinical','train'),('phantom','val'),('clinical','val'),('phantom','test')]:
        subset=[r for r in rows if r['domain']==domain and r['split']==split];lines.append(f'|{domain}/{split}|{len(subset)}|{sum(r["box"] is not None for r in subset)}|{sum(r["detector_training_overlap"] for r in subset)}|')
    lines+=['','## 完整集合准确率','', '|集合|整图|B|C|B无框回退|C无框回退|','|---|---:|---:|---:|---:|---:|']
    for name,v in results.items():lines.append('|'+name+'|'+'|'.join(f'{v[k]["accuracy"]:.2%}' for k in ['fullframe','B','C','B_fallback','C_fallback'])+'|')
    lines+=['','## 相同有框子集准确率','','|集合|有框数|整图|B|C|','|---|---:|---:|---:|---:|']
    for name,v in results.items():
        d=v['detected_only'];lines.append(f'|{name}|{d["B"]["n"]}|'+ '|'.join(f'{d[k]["accuracy"]:.2%}' for k in ['fullframe','B','C'])+'|')
    lines+=['','## 解释限制','','覆盖率不是检测召回率：真实帧没有头部框真值，预测框可能定位错误。假模验证24张被检测器训练见过，不能当作整个级联未见样本的结果；真实验证病例与训练病例分离，但参与早停，尚非最终测试集。仅单种子探索，裁剪组丢弃大量无框训练图，而且分类头不同，不能把差异完全归因于裁剪。原始图像、旧权重及实时程序未修改。', '', '详细混淆矩阵在results.json（行是真值三类，列是TooFar/Good/TooClose/Invalid）；逐帧输出见predictions.json，临床浏览见comparison_gallery.html。']
    lines+=['','## 权重与训练样本分布','','|方案|最佳轮次|实际轮次|','|---|---:|---:|']
    for arm in ['B','C']:
        history=read(OUT/arm/'history.json');best=max(history,key=lambda h:h['score']);lines.append(f'|{arm}|{best["epoch"]}|{history[-1]["epoch"]}|')
    lines+=['','有框训练样本按 TooFar/Good/TooClose 顺序：']
    for domain in ['phantom','clinical']:
        counts=Counter(r['label'] for r in rows if r['domain']==domain and r['split']=='train' and r['box'] is not None);lines.append(f'- {domain}: '+ '/'.join(str(counts[k]) for k in LABELS))
    case_results={}
    for case in sorted({r['case_id'] for r in detail if r['domain']=='clinical'}):
        subset=[r for r in detail if r['case_id']==case];case_results[case]={key:metric(subset,[r[key] for r in subset]) for key in ['fullframe','B','C','B_fallback','C_fallback']}
    dump(OUT/'clinical_by_case.json',case_results)
    lines+=['','## 真实验证按病例准确率','','|病例|整图|B纯头部|C纯头部|B回退|C回退|','|---|---:|---:|---:|---:|---:|']
    for case,metrics in case_results.items():lines.append('|'+case+'|'+'|'.join(f'{metrics[k]["accuracy"]:.2%}' for k in ['fullframe','B','C','B_fallback','C_fallback'])+'|')
    clinical=[r for r in detail if r['domain']=='clinical'];paired={}
    for arm in ['B','C']:
        eligible=[r for r in clinical if r['box'] is not None]
        paired[arm]=dict(n=len(eligible),baseline_wrong_head_right=sum(r['fullframe']!=r['label'] and r[arm]==r['label'] for r in eligible),baseline_right_head_wrong=sum(r['fullframe']==r['label'] and r[arm]!=r['label'] for r in eligible))
    dump(OUT/'paired_comparison.json',paired)
    lines+=['','同一有框真实子集上，相对整图模型的纠正/新增错误数：']
    for arm,p in paired.items():lines.append(f'- {arm}: 纠正 {p["baseline_wrong_head_right"]} 张；新增错误 {p["baseline_right_head_wrong"]} 张。')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'gallery_images').mkdir(exist_ok=True);cards=[]
    for r in detail:
        if r['domain']!='clinical':continue
        im=cv2.imread(r['image_path'])
        if r['box'] is not None:
            x1,y1,x2,y2=map(round,r['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),3)
        name=r['sample_id']+'.jpg';cv2.imwrite(str(OUT/'gallery_images'/name),base.letterbox(im,384))
        cards.append(f'<article><p>{r["sample_id"]}</p><p>真值 {r["label"]} / 整图 {r["fullframe"]}<br>B {r["B"]} / C {r["C"]}</p><img src="gallery_images/{name}"></article>')
    (OUT/'comparison_gallery.html').write_text('<meta charset="utf-8"><title>混合重训头部框对照</title><style>article{display:inline-block;width:390px;vertical-align:top;padding:8px;border:1px solid #ccc}p{overflow-wrap:anywhere}</style><h1>真实验证166帧，绿色为自动框</h1>'+''.join(cards))
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','B','C','evaluate']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='evaluate':evaluate()
    else:train(a.stage)
