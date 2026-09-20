"""Import self_zjf clinical clip labels; frozen, no-tuning domain-shift evaluation."""
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
SRC=Path(os.environ.get("NASAL_CLINICAL_ROOT", str(ROOT / "datasets/clinical_source"))).expanduser()
DATA=ROOT/'datasets/clinical_distance_self_zjf_20260918'
OUT=ROOT/'results/clinical_distance_self_zjf_20260918'
MASKCLS=ROOT/'results/head_mask_experiment_20260918/classifier'
MAPPING={'too_far':'TooFar','suitable':'Good','too_close':'TooClose'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text())
def frame_at(path,fraction):
    cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    idx=min(max(0,int(n*fraction)),max(0,n-1));cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot decode {path} frame {idx}')
    return im,idx,n,fps

def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Prepared dataset exists')
    for p in [DATA/'annotations',DATA/'clips',DATA/'frames',OUT]:p.mkdir(parents=True,exist_ok=True)
    csvpath=SRC/'annotations/self_zjf_view_state_v1.csv';jsonpath=SRC/'annotations/self_zjf/annotations.json'
    rows=list(csv.DictReader(csvpath.open(encoding='utf-8-sig')));original=read(jsonpath)
    assert len(rows)==len(original)==463
    snapshot={}
    for p in [csvpath,jsonpath,SRC/'annotations/self_zjf/config.json',SRC/'annotations/self_zjf_view_state_v1_summary.json',SRC/'nasal_endoscopy_08s/simple_08s_manifest.csv']:
        target=DATA/'annotations'/p.name;shutil.copy2(p,target);snapshot[str(p)]=sha(p);assert sha(target)==snapshot[str(p)]
    records=[];estimates=defaultdict(list);failures=[]
    for i,r in enumerate(rows):
        assert original[r['sample_key']]['annotations']['内窥镜距器械距离']==r['instrument_distance_zh']
        src=SRC/r['clip_path'];dst=DATA/'clips'/(r['clip_id']+'.mp4')
        if not src.exists():raise FileNotFoundError(src)
        shutil.copy2(src,dst);assert sha(src)==sha(dst)
        im,idx,n,fps=frame_at(dst,.5)
        try:
            fov=estimate_fov(im)
            if len(estimates[r['source_folder']])<8:estimates[r['source_folder']].append(fov)
        except ValueError as e:failures.append(dict(clip_id=r['clip_id'],reason=str(e)))
        records.append(dict(**r,label=MAPPING[r['instrument_distance']],local_clip=str(dst),clip_sha256=sha(dst),decoded_frames=n,fps=fps,shape=list(im.shape[:2])))
        if (i+1)%50==0:print('[import]',i+1,'/463',flush=True)
    calibration={}
    for source,values in estimates.items():
        calibration[source]=dict(center=np.median([v['center'] for v in values],axis=0).tolist(),diameter=float(np.median([v['diameter'] for v in values])),max_axis=float(np.median([max(v['axes']) for v in values])),n=len(values))
    frames=[]
    for i,r in enumerate(records):
        cal=calibration.get(r['source_folder'])
        for position,fraction in [('early',.25),('middle',.5),('late',.75)]:
            im,idx,n,fps=frame_at(Path(r['local_clip']),fraction);h,w=im.shape[:2]
            if cal:
                cx,cy=cal['center'];radius=.55*cal['max_axis'];x1=max(0,int(np.floor(cx-radius)));y1=max(0,int(np.floor(cy-radius)));x2=min(w,int(np.ceil(cx+radius)));y2=min(h,int(np.ceil(cy+radius)))
            else:x1,y1,x2,y2=0,0,w,h
            cropped=im[y1:y2,x1:x2];assert cropped.size
            path=DATA/'frames'/f'{r["clip_id"]}_{position}.png';assert cv2.imwrite(str(path),cropped)
            frames.append(dict(clip_id=r['clip_id'],case_id=r['case_id'],source_folder=r['source_folder'],label=r['label'],position=position,frame_index=idx,frame_time_sec=idx/fps if fps else None,image=str(path),image_sha256=sha(path),crop=[x1,y1,x2,y2],fov_diameter=cal['diameter'] if cal else None,image_quality=r['image_quality'],reviewer_note=r['reviewer_note']))
        if (i+1)%50==0:print('[frames]',i+1,'/463',flush=True)
    manifest=dict(records=records,frames=frames,source_snapshot_hashes=snapshot,calibration=calibration,fov_estimate_failures=failures,counts=dict(Counter(r['label'] for r in records)),protocol='Labels are clip-level. Middle frame primary; 25% and 75% frames secondary. Fixed optical FOV per source estimated from first up to 8 valid middle frames, no labels used. Crop center +/- 0.55 max ellipse axis, clipped to image, no resize. No clinical training or threshold tuning.')
    dump(DATA/'manifest.json',manifest)
    with (DATA/'labels.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['clip_id','case_id','source_folder','label','instrument_distance_zh','local_clip','reviewer_note'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    (DATA/'README.md').write_text('# 自标注真实手术距离验证数据\n\n来自 self_zjf，不混入expert标注。463段，16病例；65过远、296合适、102过近。clips保存原始0.8秒片段；frames保存每段25%、50%、75%帧的光学视野裁剪。annotations保留来源标注和manifest快照。\n\n原始目录只读；备注不改写。labels.csv是归一化距离标签，manifest.json记录文件哈希、采样帧、视野标定和裁剪坐标。该目录只用于验证，不加入训练集。\n')
    print('[prepared]',manifest['counts'],'frames',len(frames),'calibrated sources',len(calibration),flush=True)

@torch.inference_mode()
def evaluate():
    setup();data=read(DATA/'manifest.json')
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
                    if r['position']=='middle':cv2.imwrite(str(OUT/'gallery_images'/f'{r["clip_id"]}_{variant}.jpg'),roi)
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if r['position']=='middle':
                overlay=im.copy()
                if box is not None:
                    x1,y1,x2,y2=[int(round(v)) for v in box];cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,200,255),3)
                h,w=overlay.shape[:2];preview=cv2.resize(overlay,(640,round(h*640/w)));cv2.imwrite(str(OUT/'gallery_images'/f'{r["clip_id"]}.jpg'),preview)
                cards.append(f'<article data-label="{r["label"]}" data-error="{int(row["mask_C"]!=r["label"])}" data-invalid="{int(box is None)}"><h3>{r["clip_id"]}</h3><p>标注：{r["label"]}；RGB C：{row["rgb_C"]}；掩码 C：{row["mask_C"]}；置信度：{row["detector_confidence"]}；备注：{html.escape(r["reviewer_note"])}</p><img loading="lazy" width="540" src="gallery_images/{r["clip_id"]}.jpg">'+(''.join(f'<img loading="lazy" width="192" src="gallery_images/{r["clip_id"]}_{v}.jpg">' for v in ['rgb','mask']) if box is not None else '')+'</article>')
            if (i+1)%90==0:print('[evaluate]',i+1,'/',len(data['frames']),flush=True)
    middle=[r for r in rows if r['position']=='middle'];summary={}
    subsets={'all':middle,'far_close':[r for r in middle if r['label']!='Good'],'no_notes':[r for r in middle if not r['reviewer_note']],'clear_no_notes':[r for r in middle if not r['reviewer_note'] and r['image_quality']=='clear'],'detected_only':[r for r in middle if r['box'] is not None]}
    for name,subset in subsets.items():summary[name]={key:metrics(subset,key) for key in heads}
    # Macro-F1 on the two-label subset is restricted to supported Far/Close classes.
    for key in heads:
        m=summary['far_close'][key];cm=np.array(m['confusion']);fs=[]
        for j in [0,2]:
            tp=cm[j,j];fs.append(float(2*tp/max(2*tp+(cm[:,j].sum()-tp)+(cm[j].sum()-tp),1)))
        m['macro_f1_supported_far_close']=float(np.mean(fs))
    per_case={case:{key:metrics([r for r in middle if r['case_id']==case],key) for key in heads} for case in sorted({r['case_id'] for r in middle})}
    temporal={key:dict(early=metrics([r for r in rows if r['position']=='early'],key),late=metrics([r for r in rows if r['position']=='late'],key),all_three_predictions_identical=sum(len({x[key] for x in rows if x['clip_id']==cid})==1 for cid in {r['clip_id'] for r in middle})) for key in heads}
    summary.update(per_case=per_case,case_macro_accuracy={key:float(np.mean([v[key]['accuracy'] for v in per_case.values()])) for key in heads},temporal=temporal,coverage=dict(total=len(middle),detected=sum(r['box'] is not None for r in middle),fov_unavailable=sum(r['fov_diameter'] is None for r in middle),segmentation_empty_among_detected=sum(r['mask_empty'] is True for r in middle)),weights=hashes)
    dump(OUT/'summary.json',summary)
    with (OUT/'middle_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['clip_id','case_id','label','rgb_B','mask_B','rgb_C','mask_C','detector_confidence','box','mask_empty','fov_diameter','reviewer_note'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(middle)
    lines=['# 真实手术自标注数据：冻结模型验证','','## 数据与协议','','- 自己标注的self_zjf：463段、16病例、92视频来源；TooFar65、Good296、TooClose102。原始标注及视频已复制，未引入expert标签、未改标签、未训练或调整阈值。','- 主评估每段50%帧，前后25%/75%帧仅作时间敏感性检查，标签属于片段，未假定有精确逐帧标注。','- 每个来源视频使用前最多8个可估计中间帧的光学视野中位数标定；裁剪中心±0.55最长椭圆轴并限于原图，不缩放像素。没有根据距离标签校准FOV。','- 冻结原RGB头部检测器，conf=0.25，选择最高置信度框；RGB/掩码B/C均固定seed42权重，FP32。掩码来自当前YOLO11l，conf=0.8；背景50%亮度；1.2倍头部扩框、384等比例填充，C使用原训练尺度归一化。','- 无检测/无法提供C所需FOV时记Invalid，保留在总分母作为FN；不会当成Good。没有人工头部框，不能报告检测召回率或IoU。','','## 中间帧主结果（463段完整分母）','','|模型|准确率|Macro-F1|TooFar召回|Good召回|TooClose召回|','|---|---:|---:|---:|---:|---:|']
    for key,m in summary['all'].items():lines.append(f'|{key}|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|'+ '|'.join(f'{x:.2%}' for x in m['recall'])+'|')
    lines+=['',f'始终输出Good的多数类准确率基线：{296/463:.2%}。不要仅凭总体准确率判断远近识别。', '',f'检测覆盖：{summary["coverage"]}。这只是有预测框的比例，框是否为真实目标头部未经人工验证。','','## 过远/过近子集（167段）','','保留模型三分类输出；预测Good仍是错误。下表Macro-F1只平均Far和Close两个有真值支持的类。','','|模型|准确率|Far/Close Macro-F1|','|---|---:|---:|']
    for key,m in summary['far_close'].items():lines.append(f'|{key}|{m["accuracy"]:.2%}|{m["macro_f1_supported_far_close"]:.2%}|')
    lines+=['','## C混淆矩阵','','行TooFar/Good/TooClose；列TooFar/Good/TooClose/Invalid。']
    for key in ['rgb_C','mask_C']:lines+=['',key,'```',str(np.array(summary['all'][key]['confusion'])),'```']
    lines+=['','## 敏感性检查','','|范围|样本数|RGB C准确率|掩码C准确率|','|---|---:|---:|---:|']
    for name in ['no_notes','clear_no_notes','detected_only']:
        a=summary[name];lines.append(f'|{name}|{a["rgb_C"]["n"]}|{a["rgb_C"]["accuracy"]:.2%}|{a["mask_C"]["accuracy"]:.2%}|')
    for pos in ['early','late']:lines.append(f'|{pos}帧|463|{temporal["rgb_C"][pos]["accuracy"]:.2%}|{temporal["mask_C"][pos]["accuracy"]:.2%}|')
    lines+=['',f'16病例等权平均准确率：RGB C {summary["case_macro_accuracy"]["rgb_C"]:.2%}，掩码C {summary["case_macro_accuracy"]["mask_C"]:.2%}。逐病例结果见summary.json。','','## 解释边界','','这是冻结假模模型在自标注手术视频上的迁移验证，不是距离毫米误差验证。真实视频可能出现不同器械、多个器械、不可见头部；没有目标身份/头部框真值，不能把所有分类错误都归因于分类器。8段含备注，主结果保留并另报排除备注的敏感性结果。视频标签和具体帧可能不完全一致。','', '本批数据未用于本次头部检测器及B/C训练；原器械分割模型的训练来源与病例可能重叠，未证明整个视觉链路病例独立。来源标注与所有冻结权重进行哈希核验。数据只能称本次冻结分类/检测模型的外部迁移评估，不能据此宣称完整系统临床独立验证。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>真实手术距离验证</title><style>body{font-family:sans-serif;background:#eee;margin:24px}article{background:white;margin:16px 0;padding:12px}img{vertical-align:middle}button{margin:5px;padding:8px}</style><h1>真实手术：中间帧验证</h1><p>橙色=自动头部框（无人工框真值）。后两幅为RGB裁剪与掩码裁剪。原始标注未修改。</p><button onclick="filter(\'all\')">全部</button><button onclick="filter(\'extreme\')">过远/过近</button><button onclick="filter(\'error\')">掩码C错误</button><button onclick="filter(\'invalid\')">无检测框</button>'+''.join(cards)+'<script>function filter(mode){document.querySelectorAll("article").forEach(e=>e.hidden=mode==="all"?false:mode==="extreme"?e.dataset.label==="Good":e.dataset[mode]!=="1")}</script>')
    for p,digest in hashes.items():assert sha(p)==digest
    for p,digest in data['source_snapshot_hashes'].items():assert sha(p)==digest
    dump(OUT/'completion.json',dict(status='complete',weights=hashes,script_sha256=sha(__file__),frames=len(rows),clips=len(middle)))
    print(json.dumps({'all':summary['all'],'far_close':summary['far_close'],'coverage':summary['coverage']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','evaluate']);a=ap.parse_args();{'prepare':prepare,'evaluate':evaluate}[a.stage]()
