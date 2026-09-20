"""Validate human-confirmed exported keyframes with frozen models."""
import argparse,csv,hashlib,html,json,os,shutil,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from distance_state_classifier.src.head_roi import estimate_fov
from distance_state_classifier.src.manual_head_abc import FusionHead,make_backbone,prepare_inputs,image_tensor
from run_head_mask_experiment import CLS,OLD,SEG,soft_mask,setup
from evaluate_forceps_head_replacement import metrics,LABELS
SRC=Path(os.environ.get("NASAL_KEYFRAME_EXPORT", str(ROOT / "datasets/clinical_keyframe_export"))).expanduser()
DATA=ROOT/'datasets/clinical_keyframes_self_zjf_20260919'
OUT=ROOT/'results/clinical_keyframes_self_zjf_20260919'
MASKCLS=ROOT/'results/head_mask_experiment_20260918/classifier'
CALIBRATION=ROOT/'datasets/clinical_distance_self_zjf_20260918/manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text())

def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Prepared dataset exists')
    for p in [DATA/'source_export',DATA/'frames',OUT]:p.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((SRC/'labels.csv').open(encoding='utf-8-sig')))
    selection=read(SRC/'selections.json');export_summary=read(SRC/'summary.json');previous=read(CALIBRATION)
    assert len(rows)==export_summary['selected_frames']
    assert len({r['sample_id'] for r in rows})==len(rows)
    snapshots={};records=[];frames=[]
    for name in ['labels.csv','selected_frames.csv','selections.json','summary.json']:
        p=SRC/name;dst=DATA/'source_export'/name;shutil.copy2(p,dst);snapshots[str(p)]=sha(p);assert sha(dst)==sha(p)
    selected_lookup={(clip,int(f['frame_index'])):f for clip,entry in selection['clips'].items() for f in entry['frames']}
    for i,r in enumerate(rows):
        assert r['label'] in LABELS
        assert selected_lookup[r['clip_id'],int(r['frame_index'])]['label']==r['label']
        assert selection['clips'][r['clip_id']]['reviewed']
        src=Path(r['image_path']);rel=src.relative_to(SRC)
        dst=DATA/'source_export'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        digest=sha(src);assert sha(dst)==digest
        im=cv2.imread(str(dst));assert im is not None
        cal=previous['calibration'][r['video_id']];cx,cy=cal['center'];radius=.55*cal['max_axis'];h,w=im.shape[:2]
        shapes={tuple(x['shape']) for x in previous['records'] if x['source_folder']==r['video_id']};assert (h,w) in shapes
        x1=max(0,int(np.floor(cx-radius)));y1=max(0,int(np.floor(cy-radius)));x2=min(w,int(np.ceil(cx+radius)));y2=min(h,int(np.ceil(cy+radius)))
        cropped=im[y1:y2,x1:x2];path=DATA/'frames'/(r['sample_id']+'.png');assert cv2.imwrite(str(path),cropped)
        record={**r,'original_image_path':r['image_path'],'image_path':str(dst),'source_image_sha256':digest,'evaluation_image':str(path)};records.append(record)
        frames.append(dict(sample_id=r['sample_id'],clip_id=r['clip_id'],case_id=r['case_id'],source_folder=r['video_id'],label=r['label'],position='selected',sampling='human_selected_export',frame_index=int(r['frame_index']),frame_time_sec=float(r['clip_time_sec']),image=str(path),image_sha256=sha(path),crop=[x1,y1,x2,y2],fov_diameter=cal['diameter'],image_quality=r['clip_image_quality'],reviewer_note=' | '.join(x for x in [r['frame_note'],r['clip_note']] if x)))
        if (i+1)%100==0:print('[import]',i+1,'/',len(rows),flush=True)
    # Every item is the exact exported user-selected frame, never re-sampled.
    manifest=dict(records=records,frames=frames,source_snapshot_hashes=snapshots,calibration_source=str(CALIBRATION),calibration_sha256=sha(CALIBRATION),counts=dict(Counter(r['label'] for r in records)),cases=len({r['case_id'] for r in rows}),clips=len({r['clip_id'] for r in rows}),sources=len({r['video_id'] for r in rows}),frame_overrides_clip_label=sum(r['label']!=r['clip_distance_label'] for r in rows),protocol='Human-selected frames only, frame label overrides clip label. Reuse previous fixed optical calibration/crop and all frozen seed42 weights and thresholds. No training or tuning.')
    dump(DATA/'manifest.json',manifest)
    for name,subset in [('labels.csv',records),('labels_far_close.csv',[r for r in records if r['label']!='Good'])]:
        with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(subset)
    (DATA/'README.md').write_text('# 人工确认的真实手术关键帧验证集\n\nsource_export保存本次导出快照和原始PNG；labels.csv使用本地路径，并保留原路径；labels_far_close.csv是过远/过近子集。frames仅裁去外围黑边，不缩放，裁剪沿用20260918光学标定。全部样本采用导出的逐帧label，不用片段标签覆盖。manifest记录哈希和来源。仅做冻结模型验证，不加入训练集。\n')
    print('[prepared]',manifest['counts'],flush=True)

@torch.inference_mode()
def evaluate():
    setup();data=read(DATA/'manifest.json');assert sha(CALIBRATION)==data['calibration_sha256']
    if (OUT/'completion.json').exists():raise FileExistsError('Completed results exist')
    (OUT/'gallery_images').mkdir(exist_ok=True)
    headpaths={f'{v}_{a}':folder/f'runs/{"B_roi" if a=="B" else "C_roi_scale"}/seed_42/best.pt' for v,folder in [('rgb',CLS),('mask',MASKCLS)] for a in ['B','C']}
    weights=[SEG,OLD/'train/weights/best.pt',CLS/'frozen_backbone.pt',*headpaths.values()];hashes={str(p):sha(p) for p in weights}
    det=YOLO(str(OLD/'train/weights/best.pt'));seg=YOLO(str(SEG))
    backbone=make_backbone().cuda().eval();backbone.load_state_dict(torch.load(CLS/'frozen_backbone.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    heads={};scalers={}
    for key,path in headpaths.items():
        ckpt=torch.load(path,map_location='cpu',weights_only=True);head=FusionHead().cuda().eval();head.load_state_dict(ckpt['model_state'],strict=True);heads[key]=head;scalers[key]=(ckpt['scale_mean'].cuda(),ckpt['scale_std'].cuda())
    rows=[];cards=[]
    with (OUT/'frame_predictions.jsonl').open('w') as stream:
        for i,r in enumerate(data['frames']):
            assert sha(r['image'])==r['image_sha256'];im=cv2.imread(r['image'])
            result=det.predict(im,imgsz=1024,conf=.25,iou=.7,max_det=100,rect=True,device=0,half=False,verbose=False)[0]
            candidates=[dict(box=b,confidence=float(c)) for b,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())];chosen=max(candidates,key=lambda a:a['confidence']) if candidates else None;box=chosen['box'] if chosen else None
            row={**r,'box':box,'detector_confidence':chosen['confidence'] if chosen else None,'candidates':candidates,'mask_empty':None}
            for key in heads:row[key]='Invalid';row[key+'_probs']=None
            if box is not None:
                s=seg.predict(im,imgsz=1024,conf=.8,iou=.7,retina_masks=True,device=0,half=False,verbose=False)[0];mask=np.zeros(im.shape[:2],np.uint8)
                if s.masks is not None:mask[(s.masks.data.cpu().numpy()>.5).any(0)]=255
                row['mask_empty']=not bool(mask.any());masked=soft_mask(im,mask)
                for variant,image in [('rgb',im),('mask',masked)]:
                    _,roi,scale=prepare_inputs(image,box,r['fov_diameter'] or 1.)
                    visual=backbone(image_tensor(roi)[None].cuda())
                    for arm in ['B','C']:
                        key=variant+'_'+arm
                        if arm=='C' and r['fov_diameter'] is None:continue
                        mean,std=scalers[key];scaled=((torch.from_numpy(scale).cuda()-mean)/std)[None] if arm=='C' else torch.zeros((1,2),device='cuda')
                        probs=heads[key](visual,scaled).softmax(-1)[0].cpu().tolist();row[key]=LABELS[int(np.argmax(probs))];row[key+'_probs']=probs
                    if r['position']=='selected':cv2.imwrite(str(OUT/'gallery_images'/f'{r["sample_id"]}_{variant}.jpg'),roi)
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if r['position']=='selected':
                overlay=im.copy()
                if box is not None:
                    x1,y1,x2,y2=[int(round(v)) for v in box];cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,200,255),3)
                h,w=overlay.shape[:2];preview=cv2.resize(overlay,(640,round(h*640/w)));cv2.imwrite(str(OUT/'gallery_images'/f'{r["sample_id"]}.jpg'),preview)
                cards.append(f'<article data-label="{r["label"]}" data-error="{int(row["mask_C"]!=r["label"])}" data-invalid="{int(box is None)}"><h3>{r["sample_id"]}</h3><p>标注：{r["label"]}；RGB C：{row["rgb_C"]}；掩码 C：{row["mask_C"]}；置信度：{row["detector_confidence"]}；备注：{html.escape(r["reviewer_note"])}</p><img loading="lazy" width="540" src="gallery_images/{r["sample_id"]}.jpg">'+(''.join(f'<img loading="lazy" width="192" src="gallery_images/{r["sample_id"]}_{v}.jpg">' for v in ['rgb','mask']) if box is not None else '')+'</article>')
            if (i+1)%90==0:print('[evaluate]',i+1,'/',len(data['frames']),flush=True)
    summary={}
    subsets={'all':rows,'far_close':[r for r in rows if r['label']!='Good'],'detected_only':[r for r in rows if r['box'] is not None],'no_notes':[r for r in rows if not r['reviewer_note']]}
    for name,subset in subsets.items():summary[name]={key:metrics(subset,key) for key in heads}
    per_case={g:{key:metrics([r for r in rows if r['case_id']==g],key) for key in heads} for g in sorted({r['case_id'] for r in rows})}
    per_clip={g:{key:metrics([r for r in rows if r['clip_id']==g],key) for key in heads} for g in sorted({r['clip_id'] for r in rows})}
    summary.update(per_case=per_case,per_clip=per_clip,case_equal_weight_accuracy={key:float(np.mean([v[key]['accuracy'] for v in per_case.values()])) for key in heads},clip_equal_weight_accuracy={key:float(np.mean([v[key]['accuracy'] for v in per_clip.values()])) for key in heads},coverage=dict(total=len(rows),detected=sum(r['box'] is not None for r in rows),segmentation_empty_among_detected=sum(r['mask_empty'] is True for r in rows)),weights=hashes)
    dump(OUT/'summary.json',summary)
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['sample_id','clip_id','case_id','frame_index','label','rgb_B','mask_B','rgb_C','mask_C','detector_confidence','box','mask_empty','fov_diameter','reviewer_note'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# 人工确认关键帧：冻结模型验证','',f'共{len(rows)}张，{data["clips"]}段，{data["cases"]}病例，{data["sources"]}视频来源。类别：{data["counts"]}。采用逐帧导出标签，其中{data["frame_overrides_clip_label"]}张覆盖了原片段标签。','', '## 固定协议','', '- 本轮全部输入都是用户选择并导出的帧；不重新抽取中间帧，不沿用旧的片段标签作为逐帧真值。JSON选帧记录与导出CSV标签逐一核验。', '- 原导出完整复制至source_export；推理裁剪复用前次按来源估计的光学FOV，只去外围黑边、不缩放。没有根据新标签调整标定。', '- 原RGB头部检测器conf=0.25，最高置信度框；原/掩码B/C均seed42、FP32，分割YOLO11l conf=0.8，掩码背景50%亮度。权重、阈值和预处理保持固定，不训练。', '- 无检测输出记Invalid，保留在分母；无人工头部框，所以预测框覆盖率不是检测召回率。B无显式尺度，C有原框宽高/FOV尺度。', '', '## 全部选定帧主结果', '', '|模型|正确/总数|准确率|Macro-F1|TooFar召回|Good召回|TooClose召回|','|---|---:|---:|---:|---:|---:|---:|']
    for key,m in summary['all'].items():
        correct=sum(m['confusion'][j][j] for j in range(3));lines.append(f'|{key}|{correct}/{m["n"]}|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|'+ '|'.join(f'{x:.2%}' for x in m['recall'])+'|')
    lines+=['', f'有预测框{summary["coverage"]["detected"]}/{len(rows)}；有框时分割为空{summary["coverage"]["segmentation_empty_among_detected"]}张。始终输出Good的多数类准确率为{data["counts"]["Good"]/len(rows):.2%}。', '', '## 子集与分组结果', '', '|评估方式|样本数|RGB C准确率|掩码C准确率|','|---|---:|---:|---:|']
    for name in ['far_close','detected_only','no_notes']:
        a=summary[name];lines.append(f'|{name}|{a["rgb_C"]["n"]}|{a["rgb_C"]["accuracy"]:.2%}|{a["mask_C"]["accuracy"]:.2%}|')
    for key,n in [('clip_equal_weight_accuracy',data['clips']),('case_equal_weight_accuracy',data['cases'])]:lines.append(f'|{key}|{n}组|{summary[key]["rgb_C"]:.2%}|{summary[key]["mask_C"]:.2%}|')
    lines+=['', '过远/过近子集仍保留模型三类输出，预测Good计错。分组等权平均的是组内逐帧准确率，不是多数投票，不把同片段相邻帧当成独立病例。', '', '## C混淆矩阵', '', '行TooFar/Good/TooClose；列TooFar/Good/TooClose/Invalid。']
    for key in ['rgb_C','mask_C']:lines+=['',key,'```',str(np.array(summary['all'][key]['confusion'])),'```']
    lines+=['', '## 结果适用范围','', '此次使用人工确认的逐帧标签，可以评价这些选定帧；它与前次固定抽帧、套用片段标签的结果不是同一评估集，不能直接把指标差异解释成模型改善或恶化。前次结果仅作采样诊断。', '', '选定帧是有选择的验证集合，并不代表完整实时视频的自然分布。没有头部框/器械身份真值，分类错误可能同时来自定位、器械类型、标注定义和分类域差异；不能仅凭分类表隔离原因。', '', '本次未用这些关键帧训练头部检测器或B/C；器械分割器训练来源与病例重叠尚未核实，不能声称完整链路病例独立。原导出、原标签、旧权重和实时程序均未修改。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>人工选帧验证</title><style>body{font-family:sans-serif;background:#eee;margin:24px}article{background:white;margin:16px 0;padding:12px}img{vertical-align:middle}button{margin:5px;padding:8px}</style><h1>人工确认关键帧验证</h1><p>逐帧导出标签。橙框是自动头部框，不是人工真值。后两图为RGB与掩码裁剪。</p><button onclick="filter(\'all\')">全部</button><button onclick="filter(\'extreme\')">过远/过近</button><button onclick="filter(\'error\')">掩码C错误</button><button onclick="filter(\'invalid\')">无预测框</button>'+''.join(cards)+'<script>function filter(mode){document.querySelectorAll("article").forEach(e=>e.hidden=mode==="all"?false:mode==="extreme"?e.dataset.label==="Good":e.dataset[mode]!=="1")}</script>')
    for p,digest in hashes.items():assert sha(p)==digest
    for p,digest in data['source_snapshot_hashes'].items():assert sha(p)==digest
    dump(OUT/'completion.json',dict(status='complete',weights=hashes,script_sha256=sha(__file__),frames=len(rows),clips=data['clips'],cases=data['cases']))
    print(json.dumps({'all':summary['all'],'far_close':summary['far_close'],'coverage':summary['coverage']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate']);a=ap.parse_args();{'prepare':prepare,'evaluate':evaluate}[a.stage]()
