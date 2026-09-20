"""Paired manual/automatic ROI evaluation with frozen C seed 42; no oracle fallback."""
import csv, hashlib, html, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from distance_state_classifier.src.manual_head_abc import FusionHead, make_backbone, prepare_inputs, image_tensor
OUT=ROOT/'results/forceps_head_yolo11s_20260917'
C=ROOT/'results/convnext_manual_head_abc_20260917'
LABELS=['TooFar','Good','TooClose']
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def iou(a,b):
    if a is None or b is None:return 0.
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=x*y
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def metrics(rows,key):
    cm=np.zeros((3,4),dtype=int)
    for r in rows: cm[LABELS.index(r['label']),LABELS.index(r[key]) if r[key] in LABELS else 3]+=1
    tp=np.diag(cm[:,:3]); recall=tp/np.maximum(cm.sum(1),1);precision=tp/np.maximum(cm[:,:3].sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(n=len(rows),accuracy=float(tp.sum()/len(rows)) if rows else None,macro_f1=float(f1.mean()) if rows else None,recall=recall.tolist(),confusion=cm.tolist(),columns=LABELS+['Invalid'])
@torch.inference_mode()
def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    audit=json.loads((OUT/'data_audit.json').read_text())
    for p,digest in audit['frozen_classifier_hashes'].items():assert sha(p)==digest,p
    detector=YOLO(str(OUT/'train/weights/best.pt'))
    detmetrics=detector.val(data=str(OUT/'dataset.yaml'),split='val',imgsz=1024,batch=8,device=0,workers=0,half=False,plots=True,project=str(OUT),name='detector_validation',exist_ok=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    backbone=make_backbone().cuda().eval()
    backbone.load_state_dict(torch.load(C/'frozen_backbone.pt',map_location='cpu',weights_only=True)['model_state'],strict=True)
    ckpt=torch.load(C/'runs/C_roi_scale/seed_42/best.pt',map_location='cpu',weights_only=True)
    head=FusionHead().cuda().eval();head.load_state_dict(ckpt['model_state'],strict=True)
    mean,std=ckpt['scale_mean'].cuda(),ckpt['scale_std'].cuda()
    calibration=json.loads((ROOT/'results/convnext_head_roi_cache_20260907/audit.json').read_text())['calibration']
    old={r['sample_id']:r for r in csv.DictReader((C/'predictions.csv').open(encoding='utf-8-sig')) if r['arm']=='C_roi_scale' and r['seed']=='42' and r['split']=='val'}
    def classify(im,box,d):
        if box is None:return 'Invalid',None,None
        _,roi,scale=prepare_inputs(im,box,d)
        probs=head(backbone(image_tensor(roi)[None].cuda()),((torch.from_numpy(scale).cuda()-mean)/std)[None]).softmax(-1)[0].cpu().numpy()
        return LABELS[int(probs.argmax())],probs.tolist(),roi
    rows=[];cards=[];baseline_error=0.; images=[]
    (OUT/'comparison_images').mkdir(exist_ok=True)
    for r in audit['records']:
        if r['split']!='val':continue
        im=cv2.imread(r['image']);images.append(im);d=calibration[r['video_id']]['diameter']
        result=detector.predict(im,imgsz=1024,device=0,half=False,conf=.25,iou=.7,max_det=100,rect=True,verbose=False)[0]
        candidates=[dict(box=b,confidence=float(c)) for b,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        chosen=max(candidates,key=lambda x:x['confidence']) if candidates else None
        pred_box=chosen['box'] if chosen else None
        manual,mp,mroi=classify(im,r['box'],d);automatic,ap,aroi=classify(im,pred_box,d)
        if r['box'] is not None:
            assert r['label']==old[r['sample_id']]['label']
            prior=np.array([float(old[r['sample_id']]['p_'+x]) for x in LABELS]); baseline_error=max(baseline_error,float(np.max(np.abs(prior-mp))))
            assert manual==old[r['sample_id']]['prediction']
        row=dict(sample_id=r['sample_id'],label=r['label'],has_head=r['box'] is not None,manual_box=r['box'],auto_box=pred_box,confidence=chosen['confidence'] if chosen else None,candidates=candidates,manual_prediction=manual,auto_prediction=automatic,manual_probs=mp,auto_probs=ap,selected_iou=iou(r['box'],pred_box))
        if r['box'] is not None and pred_box is not None:
            row['width_ratio']=(pred_box[2]-pred_box[0])/(r['box'][2]-r['box'][0]);row['height_ratio']=(pred_box[3]-pred_box[1])/(r['box'][3]-r['box'][1])
        rows.append(row)
        shown=im.copy()
        for box,color in [(r['box'],(0,255,0)),(pred_box,(0,165,255))]:
            if box is not None:
                x1,y1,x2,y2=map(lambda x:int(round(x)),box);cv2.rectangle(shown,(x1,y1),(x2,y2),color,3)
        cv2.imwrite(str(OUT/'comparison_images'/f'{r["sample_id"]}_boxes.jpg'),shown)
        for kind,roi in [('manual',mroi),('auto',aroi)]:
            if roi is not None:cv2.imwrite(str(OUT/'comparison_images'/f'{r["sample_id"]}_{kind}.jpg'),roi)
        tag='无可见头部' if not row['has_head'] else r['label']
        cards.append(f'<article><h3>{html.escape(r["sample_id"])}</h3><p>标签：{tag}；人工框：{manual}；自动框：{automatic}；IoU={row["selected_iou"]:.3f}；置信度={row["confidence"]}</p><img width="560" src="comparison_images/{r["sample_id"]}_boxes.jpg">'+''.join(f'<img width="192" src="comparison_images/{r["sample_id"]}_{k}.jpg">' for k,v in [('manual',mroi),('auto',aroi)] if v is not None)+'</article>')
    assert baseline_error<1e-4,baseline_error
    positive=[r for r in rows if r['has_head']];paired=[r for r in positive if r['auto_box'] is not None];negative=[r for r in rows if not r['has_head']]
    # Benchmark actual trained detector including preprocessing, transfer and NMS, predecoded frames.
    latency={}
    for half in (False,True):
        bench=YOLO(str(OUT/'train/weights/best.pt'));times=[]
        for i in range(145):
            im=images[i%len(images)];torch.cuda.synchronize();t=time.perf_counter()
            bench.predict(im,imgsz=1024,device=0,half=half,conf=.25,iou=.7,max_det=100,rect=True,verbose=False)
            torch.cuda.synchronize()
            if i>=25:times.append((time.perf_counter()-t)*1000)
        latency['fp16' if half else 'fp32']=dict(mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),n=len(times))
        del bench
    summary=dict(detector_metrics={k:float(v) for k,v in detmetrics.results_dict.items()},detector_checkpoint_sha256=sha(OUT/'train/weights/best.pt'),manual_all=metrics(positive,'manual_prediction'),automatic_all_including_misses=metrics(positive,'auto_prediction'),manual_detected_subset=metrics(paired,'manual_prediction'),automatic_detected_subset=metrics(paired,'auto_prediction'),head_frames=len(positive),accepted_head_frames=len(paired),missed_frames=len(positive)-len(paired),selected_box_iou50_count=sum(r['selected_iou']>=.5 for r in positive),mean_selected_iou_detected=float(np.mean([r['selected_iou'] for r in paired])) if paired else None,negative_frames=len(negative),negative_false_positive_frames=sum(r['auto_box'] is not None for r in negative),manual_correct_to_auto_wrong=[r['sample_id'] for r in positive if r['manual_prediction']==r['label'] and r['auto_prediction']!=r['label']],manual_wrong_to_auto_correct=[r['sample_id'] for r in positive if r['manual_prediction']!=r['label'] and r['auto_prediction']==r['label']],baseline_max_probability_error=baseline_error,detector_latency=latency)
    for p,digest in audit['frozen_classifier_hashes'].items():assert sha(p)==digest
    dump(OUT/'comparison_predictions.json',rows);dump(OUT/'comparison_summary.json',summary)
    with (OUT/'comparison_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','label','has_head','manual_prediction','auto_prediction','confidence','selected_iou','width_ratio','height_ratio','manual_box','auto_box'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (OUT/'comparison_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>人工框 → 自动框</title><style>body{font-family:sans-serif;background:#eee;margin:24px}article{background:white;padding:16px;margin:16px 0}img{vertical-align:middle;margin:4px}</style><h1>固定 C：人工框 → 自动框</h1><p>绿色=人工框；橙色=自动框。其后依次为人工框裁剪与自动框裁剪（1.2倍扩框、384等比例填充）。检测阈值0.25，选最高置信度，不使用真值筛框；无检测=Invalid。</p>'+''.join(cards))
    a=summary['manual_all'];b=summary['automatic_all_including_misses'];m=summary['detector_metrics']
    report=f'''# 单类别头部检测器与固定 C 替换对照\n\n## 协议\n\n- YOLO11s，COCO 预训练，1024 输入；训练144张（94正、50负），验证36张（34正、2负）。完整参数见 train_config.json。\n- C 固定 seed 42；backbone、分类头、尺度归一化参数不变。人工和自动框均扩大1.2倍、等比例填充到384；尺度来自未扩大的框宽高/FOV直径。\n- 检测权重按检测验证指标选择，未按分类准确率挑选。分类主结果采用 FP32、conf=0.25、NMS IoU=0.7，直接取最高置信度框，无真值辅助、无漏检补框、无时序平滑。\n- 52张已完成空标注作为“无可见头部”检测负样本；其中42张此前明确确认，另外10张保留早期完成的空标注。无可见头部的2张验证帧只评估误检，不赋予深度分类真值。\n\n## 分类替换结果\n\n|输入|分母|准确率|Macro-F1|\n|---|---:|---:|---:|\n|人工框 → 固定 C|{a['n']}|{a['accuracy']:.2%}|{a['macro_f1']:.2%}|\n|自动框 → 固定 C（漏检计Invalid）|{b['n']}|{b['accuracy']:.2%}|{b['macro_f1']:.2%}|\n\n自动框可用 {len(paired)}/{len(positive)}；漏检 {len(positive)-len(paired)}。同一自动框可用子集：人工框准确率 {summary['manual_detected_subset']['accuracy']}，自动框准确率 {summary['automatic_detected_subset']['accuracy']}。完整分母的Macro-F1将Invalid计入对应真值类的FN。\n\n人工框混淆矩阵（行TooFar/Good/TooClose；列TooFar/Good/TooClose/Invalid）：\n```\n{np.array(a['confusion'])}\n```\n自动框混淆矩阵：\n```\n{np.array(b['confusion'])}\n```\n\n人工正确→自动错误 {len(summary['manual_correct_to_auto_wrong'])} 张；人工错误→自动正确 {len(summary['manual_wrong_to_auto_correct'])} 张。逐帧见 comparison_predictions.csv 与 comparison_gallery.html。人工框复算与原 C 保存概率最大差 {baseline_error:.3g}，两个 C 权重SHA256前后均一致。\n\n## 检测与耗时\n\n检测 mAP50={m.get('metrics/mAP50(B)',float('nan')):.4f}，mAP50–95={m.get('metrics/mAP50-95(B)',float('nan')):.4f}。固定0.25阈值下，最高置信度框IoU≥0.5为 {summary['selected_box_iou50_count']}/{len(positive)}；有预测帧平均IoU={summary['mean_selected_iou_detected']}。空标注误检 {summary['negative_false_positive_frames']}/{len(negative)}，样本很少，不能据此估计部署误检率。AP使用检测评估默认低阈值扫曲线，与固定阈值指标口径不同。\n\nRTX3090、batch1、预读图像，包含检测预处理/传输/推理/NMS，25次预热、120次计时、CUDA同步：\n\n|精度|均值 ms|P95 ms|\n|---|---:|---:|\n|FP32|{latency['fp32']['mean_ms']:.2f}|{latency['fp32']['p95_ms']:.2f}|\n|FP16|{latency['fp16']['mean_ms']:.2f}|{latency['fp16']['p95_ms']:.2f}|\n\nFP16这里只测耗时，分类主对照仍使用FP32；不代表整条视觉链路20FPS已验收。未计相机、分割/尖端、C分类、显示与机器人通信。\n\n## 局限与使用\n\n当前是人工重新混合后的同视频开发验证集，参与了检测器和原C的模型选择；不是独立测试，更不能证明真实手术泛化。分类验证仅34张，1张对应2.94个百分点。自动框分类结果包含定位误差造成的裁剪内容、尺度输入两方面变化。\n\n检测权重：train/weights/best.pt。本次没有修改实时控制程序；无检测时应输出“头部不可用”，不能默认Good或触发轴向跟随。后续接入前还需验证真实视频上的检测、尺度抖动和完整链路延迟。\n'''
    (OUT/'report.md').write_text(report)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
