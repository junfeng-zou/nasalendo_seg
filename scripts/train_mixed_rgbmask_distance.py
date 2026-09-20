"""Four-channel full-FOV ConvNeXt; fixed predicted instrument masks, unchanged splits."""
import train_mixed_fullframe_distance as base
from train_mixed_fullframe_distance import *
OUT=ROOT/'results/convnext_mixed_rgbmask_20260920'
MASKDATA=ROOT/'datasets/mixed_rgbmask_distance_20260920'
SEG=ROOT/'larger_surgical_video_dataset_training_20260712/training_results/yolo11/yolo11l_updated_20260712_223026/weights/best.pt'
def mask_letterbox(mask,size=384):
    h,w=mask.shape;ratio=min(size/w,size/h);nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
    out=np.zeros((size,size),np.uint8);x,y=(size-nw)//2,(size-nh)//2
    out[y:y+nh,x:x+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
    return out

def prepare():
    assert not (MASKDATA/'manifest.json').exists(),'Preserve existing mask cache'
    from ultralytics import YOLO
    OUT.mkdir(parents=True,exist_ok=True);(MASKDATA/'masks').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=YOLO(str(SEG));records=[]
    for i,r in enumerate(read(DATA/'manifest.json')['records']):
        assert sha(r['image_path'])==r['image_sha256'];image=cv2.imread(r['image_path']);mask=np.zeros(image.shape[:2],np.uint8)
        result=model.predict(image,imgsz=1024,conf=.8,iou=.7,retina_masks=True,device=0,half=False,verbose=False)[0]
        if result.masks is not None:
            masks=result.masks.data.cpu().numpy();assert masks.shape[1:]==mask.shape;mask[(masks>.5).any(0)]=255
        path=MASKDATA/'masks'/(r['sample_id']+'.png');assert cv2.imwrite(str(path),mask_letterbox(mask))
        records.append(dict(sample_id=r['sample_id'],mask_path=str(path),mask_sha256=sha(path),mask_empty=not bool(mask.any()),mask_area_fraction=float((mask>0).mean()),image_sha256=r['image_sha256']))
        if (i+1)%200==0:print('masks',i+1,flush=True)
    protocol=dict(imgsz=1024,conf=.8,iou=.7,retina_masks=True,half=False,selection='union of instrument instances',resize='native mask -> same RGB letterbox geometry with nearest interpolation')
    dump(MASKDATA/'manifest.json',dict(records=records,segmentation_model=str(SEG),segmentation_sha256=sha(SEG),source_manifest_sha256=sha(DATA/'manifest.json'),protocol=protocol))
    cfg=read(base.OUT/'config.json');cfg.update(in_chans=4,preprocess='Full-FOV RGB letterbox384/ImageNet normalization + aligned binary 0/1 instrument mask; empty masks retained; no head detector',mask_manifest_sha256=sha(MASKDATA/'manifest.json'),segmentation=protocol,segmentation_sha256=sha(SEG),baseline_checkpoint_sha256=sha(base.OUT/'best.pt'))
    dump(OUT/'config.json',cfg)
    print('prepared',len(records),'empty',sum(r['mask_empty'] for r in records),flush=True)

class CachedImages(base.CachedImages):
    def __init__(self,rows,augmentation=None):
        super().__init__(rows,augmentation)
        lookup={r['sample_id']:r for r in read(MASKDATA/'manifest.json')['records']}
        self.masks=[]
        for r in rows:
            entry=lookup[r['sample_id']];assert entry['image_sha256']==r['image_sha256'];assert sha(entry['mask_path'])==entry['mask_sha256']
            self.masks.append(torch.from_numpy(cv2.imread(entry['mask_path'],0).astype(np.float32)/255)[None])
    def __getitem__(self,i):
        rgb,target,index=super().__getitem__(i)
        return torch.cat([rgb,self.masks[i]],dim=0),target,index

def make_model(pretrained=True):
    model=timm.create_model('convnext_tiny',pretrained=False,num_classes=1000,drop_rate=.2)
    if pretrained:model.load_state_dict(load_file(str(PRETRAIN)),strict=True)
    model.reset_classifier(3)
    stem=model.stem[0]
    # Do not advance training RNG relative to the RGB initialization.
    state=torch.get_rng_state();conv=nn.Conv2d(4,stem.out_channels,stem.kernel_size,stem.stride,stem.padding,bias=stem.bias is not None);torch.set_rng_state(state)
    with torch.no_grad():
        conv.weight[:,:3].copy_(stem.weight);conv.weight[:,3:].zero_()
        if stem.bias is not None:conv.bias.copy_(stem.bias)
    model.stem[0]=conv
    return model

def verify_inputs():
    cfg=read(OUT/'config.json');assert cfg['mask_manifest_sha256']==sha(MASKDATA/'manifest.json');assert cfg['segmentation_sha256']==sha(SEG);assert cfg['baseline_checkpoint_sha256']==sha(base.OUT/'best.pt')

def train():
    verify_inputs()
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
    model=make_model().cuda()
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
    verify_inputs()
    report()

def report():
    verify_inputs();ours=read(OUT/'results.json');old=read(base.OUT/'results.json');complete=read(OUT/'completion.json');manifest=read(DATA/'manifest.json');masks={r['sample_id']:r for r in read(MASKDATA/'manifest.json')['records']}
    baseline={r['sample_id']:r for r in csv.DictReader((base.OUT/'predictions.csv').open(encoding='utf-8-sig'))};preds=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8-sig')))
    lines=['# 整图 RGB＋预测器械掩码四通道实验','','固定YOLO11l分割权重，预测二值器械掩码作为第四通道；保留原始RGB，不抑制背景，不使用头部检测。空掩码仍参与训练和完整验证。RGB使用ImageNet归一化，mask保持0/1。两者等比例填充384，mask用最近邻插值。新增卷积通道零初始化，RGB权重及分类头初始值与基线一致；前3轮冻结主干，第4轮起连同掩码通道一起微调。','','沿用原划分、种子42、基础颜色/模糊/噪声增强、学习率、40轮上限和10轮早停。颜色增强只作用RGB，没有几何增强。不引入额外外观增强。每轮AMP评估按两个域验证Macro-F1均值选权重，最终FP32；测试不参与选权重。', '', '## 数据与空掩码','','|域/划分|样本|空掩码|','|---|---:|---:|']
    for domain,split in [('phantom','train'),('clinical','train'),('phantom','val'),('clinical','val'),('phantom','test')]:
        selected=[r for r in manifest['records'] if r['domain']==domain and r['split']==split];lines.append(f'|{domain}/{split}|{len(selected)}|{sum(masks[r["sample_id"]]["mask_empty"] for r in selected)}|')
    lines+=['','## 完整集合对照','','|集合|RGB准确率|RGB+mask准确率|RGB Macro-F1|RGB+mask Macro-F1|','|---|---:|---:|---:|---:|']
    for name in ['clinical_val','phantom_val','phantom_test']:lines.append(f'|{name}|{old[name]["accuracy"]:.2%}|{ours[name]["accuracy"]:.2%}|{old[name]["macro_f1"]:.2%}|{ours[name]["macro_f1"]:.2%}|')
    lines+=['','## 真实验证各类召回','','|类别|RGB|RGB+mask|','|---|---:|---:|']
    for i,k in enumerate(LABELS):lines.append(f'|{k}|{old["clinical_val"]["per_class"][i]["recall"]:.2%}|{ours["clinical_val"]["per_class"][i]["recall"]:.2%}|')
    lines+=['','## 真实验证按病例','','|病例|RGB|RGB+mask|','|---|---:|---:|']
    for case,m in ours['clinical_by_case'].items():lines.append(f'|{case}|{old["clinical_by_case"][case]["accuracy"]:.2%}|{m["accuracy"]:.2%}|')
    diagnostics={}
    for empty in [True,False]:
        selected=[r for r in preds if r['domain']=='clinical' and masks[r['sample_id']]['mask_empty']==empty];diagnostics[str(empty)]=dict(n=len(selected),rgb_correct=sum(baseline[r['sample_id']]['prediction']==r['label'] for r in selected),rgbmask_correct=sum(r['prediction']==r['label'] for r in selected))
    clinical=[r for r in preds if r['domain']=='clinical'];paired=dict(corrected=sum(baseline[r['sample_id']]['prediction']!=r['label'] and r['prediction']==r['label'] for r in clinical),new_errors=sum(baseline[r['sample_id']]['prediction']==r['label'] and r['prediction']!=r['label'] for r in clinical))
    dump(OUT/'mask_diagnostics.json',dict(empty_mask_groups=diagnostics,paired=paired))
    lines+=['',f'最佳第{complete["best_epoch"]}轮，实际{complete["last_epoch"]}轮。真实验证相对RGB纠正{paired["corrected"]}张，新增错误{paired["new_errors"]}张。','','空掩码/非空掩码子集统计：','```json',json.dumps(diagnostics,ensure_ascii=False,indent=2),'```','','## 限制','','真实验证仅4个病例且用于选权重；单种子结果不是独立临床测试结论。非空掩码不代表分割正确；此处没有真实分割GT，不能报告分割IoU或召回率。冻结既有分割器，其历史训练来源与本轮数据的潜在重叠未完整审计。掩码置零诊断也不等同于重新训练的RGB模型。原RGB权重及实时程序保持不变。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');(OUT/'gallery_images').mkdir(exist_ok=True);cards=[]
    for r in clinical:
        im=cv2.imread(r['input_path']);mask=cv2.imread(masks[r['sample_id']]['mask_path'],0);overlay=im.copy();overlay[mask>0]=(.5*overlay[mask>0]+.5*np.array([0,255,0])).astype(np.uint8)
        name=r['sample_id']+'.jpg';cv2.imwrite(str(OUT/'gallery_images'/name),np.concatenate([im,overlay],axis=1));error=int(r['prediction']!=r['label'])
        cards.append(f'<article data-error="{error}"><p>{html.escape(r["sample_id"])}</p><p>真值 {r["label"]} / RGB {baseline[r["sample_id"]]["prediction"]} / RGB+mask {r["prediction"]} / 空掩码 {masks[r["sample_id"]]["mask_empty"]}</p><img loading="lazy" width="768" src="gallery_images/{name}"></article>')
    (OUT/'comparison_gallery.html').write_text('<meta charset="utf-8"><title>整图RGB+mask</title><style>article{border:1px solid #ddd;margin:12px;padding:8px;width:780px}p{overflow-wrap:anywhere}</style><h1>原图 / 预测掩码叠加；真实验证166帧</h1><button onclick="document.querySelectorAll(\'article\').forEach(e=>e.hidden=e.dataset.error!==\'1\')">仅新模型错误</button><button onclick="document.querySelectorAll(\'article\').forEach(e=>e.hidden=false)">全部</button>'+''.join(cards))
    print('Results',json.dumps({k:ours[k] for k in ['clinical_val','phantom_val','phantom_test']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','report']);args=ap.parse_args();{'prepare':prepare,'train':train,'report':report}[args.stage]()
