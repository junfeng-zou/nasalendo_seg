"""Factorial mask ablation: detector input x classifier input x B/C, three seeds."""
import csv, html, json, time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from run_head_mask_experiment import OUT,OLD,CLS,SEG,ARMS,read,dump,sha,setup,verify,soft_mask
from evaluate_forceps_head_replacement import metrics,LABELS,iou
from distance_state_classifier.src.manual_head_abc import FusionHead,make_backbone,prepare_inputs,image_tensor

def label(probs):return LABELS[int(np.argmax(probs))] if probs is not None else 'Invalid'

@torch.inference_mode()
def main():
    setup();data=verify()
    if (OUT/'completion.json').exists():raise FileExistsError('Experiment already completed')
    detector=YOLO(str(OUT/'detector_train/weights/best.pt'))
    ap=detector.val(data=str(OUT/'dataset.yaml'),split='val',imgsz=1024,batch=8,device=0,workers=0,half=False,plots=True,project=str(OUT),name='detector_validation',exist_ok=True)
    setup()
    oldpred={r['sample_id']:r for r in read(OLD/'comparison_predictions.json')}
    oldclass={(r['arm'],int(r['seed']),r['sample_id']):r for r in csv.DictReader((CLS/'predictions.csv').open(encoding='utf-8-sig'))}
    newclass={(r['arm'],r['seed'],r['sample_id']):r for r in read(OUT/'classifier/predictions.json')}
    backbone=make_backbone().cuda().eval();backbone.load_state_dict(torch.load(CLS/'frozen_backbone.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    heads={};scalers={};frozen={}
    for variant,folder in [('rgb',CLS),('mask',OUT/'classifier')]:
        for arm in ARMS:
            for seed in [42,43,44]:
                path=folder/f'runs/{arm}/seed_{seed}/best.pt';frozen[str(path)]=sha(path)
                ckpt=torch.load(path,map_location='cpu',weights_only=True)
                head=FusionHead(ckpt['config']['hidden'],ckpt['config']['dropout']).cuda().eval();head.load_state_dict(ckpt['model_state'],strict=True)
                heads[variant,arm,seed]=head;scalers[variant,arm,seed]=(ckpt['scale_mean'].cuda(),ckpt['scale_std'].cuda())
    calibration=read(ROOT/'results/convnext_head_roi_cache_20260907/audit.json')['calibration']
    rows=[];detection=[];cards=[];errors={'rgb_manual':0.,'mask_manual':0.,'rgb_auto_seed42':0.}
    (OUT/'comparison_images').mkdir(exist_ok=True)
    for r in data['records']:
        if r['split']!='val':continue
        sid=r['sample_id'];original=cv2.imread(r['source']);masked=cv2.imread(r['masked_image']);old=oldpred[sid]
        result=detector.predict(masked,imgsz=1024,conf=.25,iou=.7,device=0,half=False,rect=True,max_det=100,verbose=False)[0]
        candidates=[dict(box=b,confidence=float(c)) for b,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        chosen=max(candidates,key=lambda x:x['confidence']) if candidates else None
        boxes={'manual':r['box'],'rgb_detector':old['auto_box'],'mask_detector':chosen['box'] if chosen else None}
        d=dict(sample_id=sid,label=r['label'],has_head=r['box'] is not None,**boxes,mask_confidence=chosen['confidence'] if chosen else None,mask_candidates=candidates,mask_selected_iou=iou(r['box'],boxes['mask_detector']),mask_empty=r['mask_empty'])
        detection.append(d)
        perframe=[]
        for box_kind,box in boxes.items():
            for variant,im in [('rgb',original),('mask',masked)]:
                feature=None;scale=None
                if box is not None:
                    _,roi,scales=prepare_inputs(im,box,calibration[r['video_id']]['diameter'])
                    feature=backbone(image_tensor(roi)[None].cuda());scale=torch.from_numpy(scales).cuda()
                    cv2.imwrite(str(OUT/'comparison_images'/f'{sid}_{box_kind}_{variant}.jpg'),roi)
                for arm in ARMS:
                    for seed in [42,43,44]:
                        probs=None
                        if feature is not None:
                            mean,std=scalers[variant,arm,seed]
                            s=((scale-mean)/std)[None] if arm=='C_roi_scale' else torch.zeros((1,2),device='cuda')
                            probs=heads[variant,arm,seed](feature,s).softmax(-1)[0].cpu().tolist()
                        pred=label(probs)
                        row=dict(sample_id=sid,label=r['label'],has_head=d['has_head'],box_source=box_kind,classifier_input=variant,arm=arm,seed=seed,prediction=pred,probs=probs,box=box)
                        rows.append(row);perframe.append(row)
                        if box_kind=='manual' and d['has_head']:
                            key=(arm,seed,sid)
                            baseline=oldclass[key] if variant=='rgb' else newclass[key]
                            expected=[float(baseline['p_'+c]) for c in LABELS] if variant=='rgb' else baseline['probs']
                            err=float(np.max(np.abs(np.array(probs)-expected)));errors[variant+'_manual']=max(errors[variant+'_manual'],err)
                            assert pred==baseline['prediction'] and err<1e-4
                        if variant=='rgb' and box_kind=='rgb_detector' and arm=='C_roi_scale' and seed==42 and probs is not None:
                            err=float(np.max(np.abs(np.array(probs)-old['auto_probs'])));errors['rgb_auto_seed42']=max(errors['rgb_auto_seed42'],err);assert err<1e-4
        overlay=original.copy()
        for kind,color in [('manual',(0,255,0)),('rgb_detector',(0,165,255)),('mask_detector',(255,255,0))]:
            b=boxes[kind]
            if b is not None:
                x1,y1,x2,y2=[int(round(x)) for x in b];cv2.rectangle(overlay,(x1,y1),(x2,y2),color,3)
        cv2.imwrite(str(OUT/'comparison_images'/f'{sid}_boxes.jpg'),overlay)
        table='<table><tr><th>框来源</th><th>分类输入</th><th>B</th><th>C</th></tr>'
        for box_kind in boxes:
            for variant in ['rgb','mask']:
                vals=[next(x['prediction'] for x in perframe if x['box_source']==box_kind and x['classifier_input']==variant and x['arm']==a and x['seed']==42) for a in ARMS]
                table+=f'<tr><td>{box_kind}</td><td>{variant}</td><td>{vals[0]}</td><td>{vals[1]}</td></tr>'
        table+='</table>'
        cards.append(f'<article><h3>{sid}</h3><p>真值：{r["label"] if d["has_head"] else "无可见头部（不评估深度类别）"}；分割为空={r["mask_empty"]}；新检测框IoU={d["mask_selected_iou"]:.3f}</p>{table}<img width="620" src="comparison_images/{sid}_boxes.jpg"><img width="620" src="preview/{sid}.jpg"></article>')
    summaries=[]
    for box_kind in ['manual','rgb_detector','mask_detector']:
        for variant in ['rgb','mask']:
            for arm in ARMS:
                seed_scores=[]
                for seed in [42,43,44]:
                    chosen=[r for r in rows if r['has_head'] and r['box_source']==box_kind and r['classifier_input']==variant and r['arm']==arm and r['seed']==seed]
                    score=metrics(chosen,'prediction');valid=[r for r in chosen if r['box'] is not None]
                    subset=metrics(valid,'prediction');seed_scores.append(dict(seed=seed,all=score,detected_subset=subset))
                summaries.append(dict(box_source=box_kind,classifier_input=variant,arm=arm,seed_results=seed_scores,accuracy_mean=float(np.mean([x['all']['accuracy'] for x in seed_scores])),accuracy_std=float(np.std([x['all']['accuracy'] for x in seed_scores],ddof=1)),f1_mean=float(np.mean([x['all']['macro_f1'] for x in seed_scores]))))
    # Common-detection subset isolates crop changes from different detector coverage.
    common={d['sample_id'] for d in detection if d['has_head'] and d['rgb_detector'] is not None and d['mask_detector'] is not None}
    common_scores=[]
    for entry in summaries:
        subset=[r for r in rows if r['sample_id'] in common and r['box_source']==entry['box_source'] and r['classifier_input']==entry['classifier_input'] and r['arm']==entry['arm'] and r['seed']==42]
        common_scores.append({k:entry[k] for k in ['box_source','classifier_input','arm']}|{'metrics':metrics(subset,'prediction')})
    positive=[d for d in detection if d['has_head']];negative=[d for d in detection if not d['has_head']]
    detection_summary=dict(metrics={k:float(v) for k,v in ap.results_dict.items()},head_frames=len(positive),accepted=sum(d['mask_detector'] is not None for d in positive),missed=[d['sample_id'] for d in positive if d['mask_detector'] is None],selected_iou50=sum(d['mask_selected_iou']>=.5 for d in positive),negative_frames=len(negative),false_positive_negative_frames=sum(d['mask_detector'] is not None for d in negative))
    dump(OUT/'comparison_predictions.json',rows);dump(OUT/'detection_predictions.json',detection)
    with (OUT/'comparison_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(classification=summaries,detection=detection_summary,common_detection_subset=common_scores,baseline_probability_errors=errors)
    dump(OUT/'comparison_summary.json',summary)
    lines=['# 器械预测掩码引导实验：头部检测与 B/C 分类','', '## 本轮检验的具体方法','', '- 输入公式：`I_mask = round(I_RGB × (0.5 + 0.5 × M))`，M为冻结器械分割模型的预测二值掩码；器械不变、背景保留50%亮度。不是四通道融合，也不是硬抠除背景。','- 分割使用当前实时程序的YOLO11l权重，1024、conf=0.8、IoU=0.7，retina_masks=True取得原图坐标掩码，合并所有器械实例，不借助头部真值选实例，不使用时序跟踪。','- 检测器与之前保持同一COCO初始化、YOLO11s结构、超参数、144/36划分。只将输入换成上面的掩码引导图。','- 分类器沿用同一冻结ConvNeXt backbone、相同分类头初始化、训练超参数和94/34划分；B/C各训练seed42/43/44。不继续微调原B/C权重，确保训练起点一致。','- B=裁剪像素，无显式尺度；C=相同裁剪像素+未扩框宽高/FOV直径。两者均1.2倍扩框、384等比例填充。','- 分类器训练使用人工框；测试同时比较人工框、原RGB检测器框、新掩码检测器框。两种检测器都使用conf=0.25，最高置信度，无真值补漏检。','- 只改检测器/只改分类器/两者都改均列出，避免把联合结果误当作单一模块的改进。','', '## 检测结果','', f'新检测器mAP50={detection_summary["metrics"]["metrics/mAP50(B)"]:.2%}，mAP50–95={detection_summary["metrics"]["metrics/mAP50-95(B)"]:.2%}。固定阈值检出{detection_summary["accepted"]}/34，最高置信度框IoU≥0.5为{detection_summary["selected_iou50"]}/34；空标注帧误检{detection_summary["false_positive_negative_frames"]}/2。', '', '原RGB检测器：mAP50=87.73%，mAP50–95=42.67%；固定阈值检出30/34，IoU≥0.5为27/34，空标注误检0/2。','', '## 分类主结果：seed42，34张分母含漏检','', '|框来源|分类输入|模型|正确/总数|准确率|Macro-F1|检出子集准确率|','|---|---|---|---:|---:|---:|---:|']
    for e in summaries:
        m=e['seed_results'][0]['all'];sub=e['seed_results'][0]['detected_subset'];correct=sum(m['confusion'][i][i] for i in range(3))
        lines.append(f'|{e["box_source"]}|{e["classifier_input"]}|{e["arm"]}|{correct}/34|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|{sub["accuracy"]:.2%} ({sub["n"]}张)|')
    lines+=['', '## 三随机种子（分类头）的均值与样本标准差','', '检测器只有固定seed42；此处不是三个独立检测器实验。','', '|框来源|分类输入|模型|准确率均值 ± SD|Macro-F1均值|','|---|---|---|---:|---:|']
    for e in summaries:lines.append(f'|{e["box_source"]}|{e["classifier_input"]}|{e["arm"]}|{e["accuracy_mean"]:.2%} ± {e["accuracy_std"]:.2%}|{e["f1_mean"]:.2%}|')
    lines+=['', '## 两种检测器共同检出的相同子集（seed42）','', '|框来源|分类输入|模型|样本数|准确率|','|---|---|---|---:|---:|']
    for e in common_scores:lines.append(f'|{e["box_source"]}|{e["classifier_input"]}|{e["arm"]}|{e["metrics"]["n"]}|{e["metrics"]["accuracy"]:.2%}|')
    lines+=['', '## 解释边界','', f'原/新人工框分类基线与先前RGB自动框分类已复核，概率最大误差：{errors}。', '', '所有掩码均为同一冻结分割模型的预测，没有人工修正。180帧中5帧预测掩码为空；其中2帧人工标有头部（训练和验证各1帧）。空掩码输入表现为整图50%亮度，并未回填真值。', '', '这是同视频混合开发验证，检测器和分类器均用该验证集选择权重；分割模型的训练来源可能与这些图像重叠，未建立整条链路独立测试。这些结果不能证明真实手术泛化。只检验固定50%背景保留的这一种掩码使用方式，负面结果不意味着所有掩码融合方式无效。', '', '本次未改实时控制代码。分割为空、检测为空及错误框应分别处理，不能默认Good。完整链路含分割、检测、分类的20FPS要求仍需独立实测，不将离线缓存速度当作实时速度。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'comparison_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>掩码实验 B/C</title><style>body{font-family:sans-serif;background:#eee;margin:24px}article{background:white;padding:16px;margin:16px 0}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #aaa}img{vertical-align:middle}</style><h1>掩码实验：逐帧主种子42结果</h1><p>绿色=人工框；橙色=原RGB检测器；青色=掩码检测器。第二幅图依次为原图、掩码、软背景抑制输入。mask分类器在掩码引导图上重新训练，RGB分类器是原权重。</p>'+''.join(cards))
    verify()
    for path,digest in frozen.items():assert sha(path)==digest
    dump(OUT/'completion.json',dict(status='complete',frozen_classifier_hashes=frozen,detector_sha256=sha(OUT/'detector_train/weights/best.pt'),code_hashes={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/run_head_mask_experiment.py']},summary=summary))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
