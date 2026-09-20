"""Full-field ConvNeXt training on phantom + case-held-out clinical keyframes."""
import argparse,csv,hashlib,html,itertools,json,random,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import timm
from safetensors.torch import load_file
from distance_state_classifier.src.manual_head_abc import letterbox,image_tensor
from distance_state_classifier.src.transforms import augment_image
from distance_state_classifier_endodac.src.metrics import confusion_matrix,classification_metrics
OUT=ROOT/'results/convnext_mixed_fullframe_20260919'
DATA=ROOT/'datasets/mixed_fullframe_distance_20260919'
CLINICAL=ROOT/'datasets/clinical_keyframes_self_zjf_20260919/manifest.json'
LABELS=['TooFar','Good','TooClose']
PRETRAIN=ROOT/'weights/convnext_tiny.in12k_ft_in1k.safetensors'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def read(p):return json.loads(p.read_text())
def write_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def select_cases(rows):
    cases=sorted({r['case_id'] for r in rows});counts=Counter(r['label'] for r in rows);candidates=[]
    for group in itertools.combinations(cases,4):
        v=Counter(r['label'] for r in rows if r['case_id'] in group)
        if any(v[k]<10 or counts[k]-v[k]<10 for k in LABELS):continue
        score=sum(((v[k]-.2*counts[k])/(.2*counts[k]))**2 for k in LABELS)
        candidates.append((score,group))
    return min(candidates)[1]
def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Split is already frozen')
    for p in [DATA/'inputs',OUT]:p.mkdir(parents=True,exist_ok=True)
    clinical=read(CLINICAL);val_cases=select_cases(clinical['records'])
    assert val_cases==('case_0002','case_0008','case_0010','case_0014')
    rows=[];sources={str(CLINICAL):sha(CLINICAL)}
    for split in ['train','val','test']:
        path=ROOT/f'auto_labeling_project/data/final_dataset/{split}_labels.csv';sources[str(path)]=sha(path)
        for r in csv.DictReader(path.open(encoding='utf-8-sig')):
            rows.append(dict(sample_id='phantom__'+r['sample_id'],domain='phantom',split=split,label=r['label'],case_id='',video_id=r['video_id'],clip_id='',image_path=r['image_path']))
    for r in clinical['records']:
        rows.append(dict(sample_id='clinical__'+r['sample_id'],domain='clinical',split='val' if r['case_id'] in val_cases else 'train',label=r['label'],case_id=r['case_id'],video_id=r['video_id'],clip_id=r['clip_id'],image_path=r['evaluation_image']))
    assert len({r['sample_id'] for r in rows})==len(rows)==2077
    hashes={};cross_split=[]
    for i,r in enumerate(rows):
        assert r['label'] in LABELS
        image=cv2.imread(r['image_path']);assert image is not None
        r['image_sha256']=sha(r['image_path'])
        pixelhash=hashlib.sha256(image.tobytes()).hexdigest();r['pixel_sha256']=pixelhash
        if pixelhash in hashes and hashes[pixelhash]!=r['split']:cross_split.append(r['sample_id'])
        hashes[pixelhash]=r['split']
        path=DATA/'inputs'/(r['sample_id']+'.png');assert cv2.imwrite(str(path),letterbox(image,384))
        r['input_path']=str(path);r['input_sha256']=sha(path)
        if (i+1)%300==0:print('[prepare]',i+1,'/',len(rows),flush=True)
    assert not cross_split,('Exact decoded-image leakage',cross_split)
    for group in ['case_id','clip_id']:
        a={r[group] for r in rows if r['domain']=='clinical' and r['split']=='train'};b={r[group] for r in rows if r['domain']=='clinical' and r['split']=='val'};assert not a&b
    phantom_vids={s:{r['video_id'] for r in rows if r['domain']=='phantom' and r['split']==s} for s in ['train','val','test']}
    assert not phantom_vids['train']&phantom_vids['val'] and not phantom_vids['train']&phantom_vids['test'] and not phantom_vids['val']&phantom_vids['test']
    counts={f'{d}_{s}':dict(Counter(r['label'] for r in rows if r['domain']==d and r['split']==s)) for d in ['phantom','clinical'] for s in ['train','val','test']}
    plan=dict(records=rows,clinical_validation_cases=list(val_cases),counts=counts,source_hashes=sources,split_rule='Before training select 4 of 16 clinical cases minimizing sum of squared relative deviation from 20% per-class frame counts; require >=10/class in both splits; lexicographic tie break. No model outcomes used.',leakage_checks=dict(case_overlap=0,clip_overlap=0,cross_split_decoded_pixel_overlap=0,phantom_video_overlap=0))
    dump(DATA/'manifest.json',plan)
    for split in ['train','val','test']:write_csv(DATA/(split+'.csv'),[r for r in rows if r['split']==split])
    cfg=dict(model='convnext_tiny',initialization=str(PRETRAIN),initialization_sha256=sha(PRETRAIN),classes=LABELS,input_size=384,preprocess='Full FOV RGB -> aspect-preserving black letterbox384 -> ImageNet normalization. No head detector, instrument segmentation, ROI or scale feature.',seed=42,epochs=40,patience=10,batch_size=16,num_workers=2,freeze_backbone_epochs=3,head_lr=3e-4,backbone_lr=1e-5,weight_decay=1e-4,dropout=.2,amp=True,selection='Mean of clinical validation macro-F1 and phantom validation macro-F1; earliest epoch wins ties',augmentation=dict(enabled=True,brightness=.15,contrast=.15,saturation=.10,hue=.03,blur_prob=.15,noise_prob=.15,rotate_deg=0.,translate_frac=0.,scale_frac=0.,horizontal_flip_prob=0.),sampler='shuffle all training frames once per epoch',loss='Cross entropy weighted by inverse training class frequency, normalized mean1',split_manifest_sha256=sha(DATA/'manifest.json'))
    dump(OUT/'config.json',cfg)
    (DATA/'README.md').write_text('# 混合整图分类固定划分\n\n1614训练图=956假模+658手术帧。假模验证175、测试122；真实验证166。真实验证病例case_0002/0008/0010/0014，与训练病例和片段无交集。仅使用逐帧确认标签。源图像只读，inputs为完整视野的384等比例填充缓存。训练前已固定划分，详见manifest.json。\n')
    print('[prepared]',counts,flush=True)

class CachedImages(Dataset):
    def __init__(self,rows,augmentation=None):
        self.rows=rows;self.images=[cv2.imread(r['input_path']) for r in rows];self.augmentation=augmentation
        assert all(x is not None for x in self.images)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        image=self.images[i]
        if self.augmentation:
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);rgb=augment_image(rgb,self.augmentation);image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        return image_tensor(image),LABELS.index(self.rows[i]['label']),i

def seed_worker(worker):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed);cv2.setNumThreads(1)

def measure(model,loader,amp):
    model.eval();truth=[];probabilities=[]
    with torch.inference_mode():
        for images,targets,_ in loader:
            with torch.cuda.amp.autocast(enabled=amp):probs=model(images.cuda(non_blocking=True)).softmax(-1)
            probabilities.extend(probs.float().cpu().tolist());truth.extend(targets.tolist())
    pred=np.argmax(probabilities,axis=1).tolist();cm=confusion_matrix(truth,pred,3);m=classification_metrics(cm,LABELS)
    m.update(n=len(truth),confusion_matrix=cm.tolist());return m,probabilities

def train():
    if (OUT/'best.pt').exists():raise FileExistsError('Training output exists; preserve the run')
    assert torch.cuda.is_available(),'GPU required'
    cfg=read(OUT/'config.json');data=read(DATA/'manifest.json');assert sha(DATA/'manifest.json')==cfg['split_manifest_sha256']
    assert sha(PRETRAIN)==cfg['initialization_sha256']
    for p,h in data['source_hashes'].items():assert sha(p)==h
    for r in data['records']:assert sha(r['input_path'])==r['input_sha256']
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    train_rows=[r for r in data['records'] if r['split']=='train']
    trainset=CachedImages(train_rows,cfg['augmentation']);generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(trainset,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True,worker_init_fn=seed_worker,generator=generator,persistent_workers=True)
    subsets={f'{domain}_{split}':[r for r in data['records'] if r['domain']==domain and r['split']==split] for domain,split in [('phantom','val'),('clinical','val'),('phantom','test')]}
    eval_loaders={name:DataLoader(CachedImages(rows),batch_size=16,shuffle=False,num_workers=0,pin_memory=True) for name,rows in subsets.items()}
    model=timm.create_model('convnext_tiny',pretrained=False,num_classes=1000,drop_rate=cfg['dropout']);model.load_state_dict(load_file(str(PRETRAIN)),strict=True);model.reset_classifier(3);model=model.cuda()
    head_ids={id(p) for p in model.get_classifier().parameters()}
    optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in head_ids],lr=cfg['backbone_lr']),dict(params=[p for p in model.parameters() if id(p) in head_ids],lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs']);scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    counts=np.bincount([LABELS.index(r['label']) for r in train_rows],minlength=3);weights=counts.sum()/counts;weights=weights/weights.mean();criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'))
    history=[];best=-1;stale=0;begin=time.time()
    for epoch in range(1,cfg['epochs']+1):
        frozen=epoch<=cfg['freeze_backbone_epochs']
        for p in model.parameters():p.requires_grad=(id(p) in head_ids) or not frozen
        model.train();losses=[]
        for step,(images,targets,_) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg['amp']):loss=criterion(model(images.cuda(non_blocking=True)),targets.cuda(non_blocking=True))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        metrics={name:measure(model,l,cfg['amp'])[0] for name,l in eval_loaders.items() if name.endswith('_val')}
        score=np.mean([v['macro_f1'] for v in metrics.values()]);scheduler.step()
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=metrics,selection_score=float(score),seconds=time.time()-begin))
        if score>best+1e-12:
            best=float(score);stale=0
            payload=dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,config=cfg,classes=LABELS,model_name='timm:convnext_tiny',input_size=384,validation=metrics,selection_score=best,preprocess=cfg['preprocess'])
            torch.save(payload,OUT/'best.pt')
        else:stale+=1
        dump(OUT/'history.json',history)
        print(f'[epoch {epoch}] loss={np.mean(losses):.4f} phantom_acc={metrics["phantom_val"]["accuracy"]:.4f} clinical_acc={metrics["clinical_val"]["accuracy"]:.4f} mean_f1={score:.4f} best={best:.4f} stale={stale}',flush=True)
        if epoch>cfg['freeze_backbone_epochs'] and stale>=cfg['patience']:break
    model.load_state_dict(torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    results={};predictions=[]
    for name,l in eval_loaders.items():
        m,probs=measure(model,l,False);results[name]=m
        for r,p in zip(subsets[name],probs):predictions.append(dict(**r,evaluation=name,prediction=LABELS[int(np.argmax(p))],**{'p_'+k:float(v) for k,v in zip(LABELS,p)}))
    # FP32 final evaluation; checkpoint selection used AMP consistently each epoch.
    clinical=[r for r in predictions if r['domain']=='clinical'];case_metrics={}
    for case in sorted({r['case_id'] for r in clinical}):
        selected=[r for r in clinical if r['case_id']==case];cm=confusion_matrix([LABELS.index(r['label']) for r in selected],[LABELS.index(r['prediction']) for r in selected],3)
        case_metrics[case]=classification_metrics(cm,LABELS)|dict(n=len(selected),confusion_matrix=cm.tolist())
    results['clinical_by_case']=case_metrics;results['clinical_case_equal_accuracy']=float(np.mean([m['accuracy'] for m in case_metrics.values()]))
    dump(OUT/'results.json',results);write_csv(OUT/'predictions.csv',predictions)
    for p,h in data['source_hashes'].items():assert sha(p)==h
    for r in data['records']:assert sha(r['image_path'])==r['image_sha256']
    dump(OUT/'completion.json',dict(status='complete',best_epoch=payload['epoch'],last_epoch=epoch,seconds=time.time()-begin,gpu=torch.cuda.get_device_name(),split_sha256=sha(DATA/'manifest.json'),checkpoint_sha256=sha(OUT/'best.pt'),source_code_sha256=sha(__file__)))
    report()

def report():
    cfg=read(OUT/'config.json');data=read(DATA/'manifest.json');results=read(OUT/'results.json');complete=read(OUT/'completion.json')
    lines=['# 假模＋真实手术：整图 ConvNeXt 距离分类','', '## 输入与训练','', '- 输入为完整光学视野RGB，等比例填充384并ImageNet归一化；不使用头部框、头部检测、器械分割、掩码或额外尺度输入。','- ConvNeXt-Tiny使用本地ImageNet预训练初始化，先冻结主干3轮训练分类头，再以小学习率微调整个网络；不是只训练冻结特征的A/B/C小头实验。','- 随机种子42；逆训练类频率加权交叉熵；颜色、模糊和噪声增强；不做随机缩放/裁剪，避免破坏距离尺度。', '', '## 训练前固定数据划分','', '|来源|训练|验证|测试|','|---|---:|---:|---:|','|假模|956|175|122|','|真实选定帧|658|166|未另设|', '', '真实验证病例：case_0002、case_0008、case_0010、case_0014；与训练病例和片段无交集。选择规则仅参考病例类别数量，目标每类约20%，不参考模型预测。验证分布TooFar/Good/TooClose=40/66/60。','', '假模沿用原CSV按视频划分；训练6个视频、验证bend_data1、测试bend_data2。没有沿用此前人工框94/34的小子集。解码图像哈希核验无跨划分完全相同图像。','', '## 评估结果','', '|数据|样本数|准确率|Macro-F1|TooFar召回|Good召回|TooClose召回|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ['clinical_val','phantom_val','phantom_test']:
        m=results[name];lines.append(f'|{name}|{m["n"]}|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|'+ '|'.join(f'{p["recall"]:.2%}' for p in m['per_class'])+'|')
    lines+=['', f'最佳第{complete["best_epoch"]}轮，实际训练{complete["last_epoch"]}轮。按假模与真实验证Macro-F1的等权平均选择权重，不以混合样本总准确率选择。测试集不参与早停或权重选择。最终表格采用FP32无时序平滑、argmax，不使用0.55置信度拒绝。','', '## 真实验证混淆矩阵','', '行/列均TooFar、Good、TooClose。','```',str(np.array(results['clinical_val']['confusion_matrix'])),'```', '', '## 真实验证按病例','', '|病例|样本数|准确率|Macro-F1|','|---|---:|---:|---:|']
    for case,m in results['clinical_by_case'].items():lines.append(f'|{case}|{m["n"]}|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|')
    lines+=['', f'病例等权平均准确率：{results["clinical_case_equal_accuracy"]:.2%}。某些病例只有一个真值类别，逐病例Macro-F1仍固定平均3类，因此主要看该病例准确率和样本数。','', '## 解释边界','', '这是真实病例留出的开发验证，但验证集参与模型选择，不是未见的真实临床最终测试集。旧头部定位方案曾在全部824真实帧上做过诊断，这些病例并非从未分析过。','', '与前次824帧的头部检测＋C结果不同：此次训练数据、验证病例、输入和模型训练方式都改变，不能把差异全归因于加入真实数据。未做相同训练方案的假模单域重训消融。','', '权重与类别映射、预处理保存在best.pt/config.json。实时推理需使用本轮整图等比例填充预处理，不能直接套用旧的拉伸或头部裁剪预处理。本次没有替换实时程序。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    preds=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8-sig')));(OUT/'gallery_images').mkdir(exist_ok=True);cards=[]
    for r in preds:
        if r['domain']!='clinical':continue
        image=cv2.imread(r['input_path']);name=r['sample_id']+'.jpg';cv2.imwrite(str(OUT/'gallery_images'/name),image)
        cards.append(f'<article data-error="{int(r["label"]!=r["prediction"])}"><h3>{r["sample_id"]}</h3><p>真值 {r["label"]} → 预测 {r["prediction"]}</p><img src="gallery_images/{name}"></article>')
    (OUT/'validation_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>混合整图模型真实验证</title><style>body{font-family:sans-serif;background:#eee}article{display:inline-block;vertical-align:top;background:white;margin:10px;padding:10px;max-width:400px}h3{word-break:break-all}img{width:384px}</style><h1>真实病例留出验证：整图输入</h1><button onclick="document.querySelectorAll(\'article\').forEach(e=>e.hidden=e.dataset.error!==\'1\')">仅错误</button><button onclick="document.querySelectorAll(\'article\').forEach(e=>e.hidden=false)">全部</button>'+''.join(cards))
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','report']);args=ap.parse_args();{'prepare':prepare,'train':train,'report':report}[args.stage]()
