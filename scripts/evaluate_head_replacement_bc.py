"""Replay identical saved detector boxes through frozen B and C, seed 42."""
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from distance_state_classifier.src.manual_head_abc import FusionHead, make_backbone, prepare_inputs, image_tensor
from evaluate_forceps_head_replacement import metrics, LABELS

SOURCE = ROOT / 'results/forceps_head_yolo11s_20260917'
CLASSIFIER = ROOT / 'results/convnext_manual_head_abc_20260917'
OUT = ROOT / 'results/forceps_head_replacement_bc_20260918'
ARMS = {'B': 'B_roi', 'C': 'C_roi_scale'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


@torch.inference_mode()
def main():
    assert torch.cuda.is_available(), 'CUDA required; do not silently fall back to CPU'
    if (OUT / 'completion.json').exists():
        raise FileExistsError('Completed output exists; preserve it and select another OUT for a new run')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    boxes_file = SOURCE / 'comparison_predictions.json'
    saved = json.loads(boxes_file.read_text())
    audit = json.loads((SOURCE / 'data_audit.json').read_text())
    records = {r['sample_id']: r for r in audit['records'] if r['split'] == 'val'}
    assert len(saved) == len(records) == 36
    assert {r['sample_id'] for r in saved} == set(records)
    prepared = {r['sample_id']: r for r in json.loads((CLASSIFIER / 'prepared.json').read_text())['records'] if r['split'] == 'val'}
    old = {(r['arm'], r['sample_id']): r for r in csv.DictReader((CLASSIFIER / 'predictions.csv').open(encoding='utf-8-sig')) if r['seed'] == '42' and r['split'] == 'val'}
    ckpt_paths = {arm: CLASSIFIER / f'runs/{name}/seed_42/best.pt' for arm, name in ARMS.items()}
    frozen_paths = [CLASSIFIER / 'frozen_backbone.pt', *ckpt_paths.values(), boxes_file, SOURCE / 'train/weights/best.pt', CLASSIFIER / 'predictions.csv', CLASSIFIER / 'prepared.json']
    hashes = {str(p): sha(p) for p in frozen_paths}
    for p, digest in audit['frozen_classifier_hashes'].items():
        assert sha(p) == digest
    for r in records.values():
        assert sha(r['image']) == r['image_sha256']
        assert sha(r['annotation']) == r['annotation_sha256']
    backbone = make_backbone().cuda().eval()
    backbone.load_state_dict(torch.load(CLASSIFIER / 'frozen_backbone.pt', map_location='cpu', weights_only=True)['model_state'], strict=True)
    heads, checkpoints = {}, {}
    for arm, path in ckpt_paths.items():
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        assert ckpt['arm'] == ARMS[arm] and ckpt['seed'] == 42
        assert ckpt['config']['input_size'] == 384 and ckpt['config']['context_factor'] == 1.2
        head = FusionHead(ckpt['config']['hidden'], ckpt['config']['dropout']).cuda().eval()
        head.load_state_dict(ckpt['model_state'], strict=True)
        checkpoints[arm], heads[arm] = ckpt, head
    calibration = json.loads((ROOT / 'results/convnext_head_roi_cache_20260907/audit.json').read_text())['calibration']
    mean = checkpoints['C']['scale_mean'].cuda()
    std = checkpoints['C']['scale_std'].cuda()
    output_rows, cards = [], []
    max_error = {'B_manual_vs_original': 0., 'C_manual_vs_original': 0., 'C_auto_vs_previous': 0.}
    for prior in saved:
        sid = prior['sample_id']
        record = records[sid]
        assert prior['manual_box'] == record['box'] and prior['label'] == record['label']
        if prior['has_head']:
            assert prior['manual_box'] == prepared[sid]['box']
            assert prior['label'] == prepared[sid]['label']
        candidates = prior['candidates']
        selected = max(candidates, key=lambda x: x['confidence']) if candidates else None
        assert prior['auto_box'] == (selected['box'] if selected else None)
        image = cv2.imread(record['image'])
        diameter = calibration[record['video_id']]['diameter']
        row = {k: prior[k] for k in ['sample_id', 'label', 'has_head', 'manual_box', 'auto_box', 'confidence', 'selected_iou']}
        for kind in ['manual', 'auto']:
            box = row[kind + '_box']
            if box is None:
                for arm in ARMS:
                    row[f'{arm}_{kind}_prediction'] = 'Invalid'
                    row[f'{arm}_{kind}_probs'] = None
                continue
            _, roi, scales = prepare_inputs(image, box, diameter, 384, 1.2)
            visual = backbone(image_tensor(roi)[None].cuda())
            # B was trained with two constant zeros: never substitute C's scales.
            inputs = {'B': torch.zeros((1, 2), device='cuda'),
                      'C': ((torch.from_numpy(scales).cuda() - mean) / std)[None]}
            for arm, head in heads.items():
                probs = head(visual, inputs[arm]).softmax(-1)[0].cpu().numpy()
                prediction = LABELS[int(probs.argmax())]
                row[f'{arm}_{kind}_prediction'] = prediction
                row[f'{arm}_{kind}_probs'] = probs.tolist()
                expected = None
                if kind == 'manual':
                    baseline = old[ARMS[arm], sid]
                    assert baseline['prediction'] == prediction
                    expected = [float(baseline['p_' + label]) for label in LABELS]
                    key = arm + '_manual_vs_original'
                elif arm == 'C':
                    assert prior['auto_prediction'] == prediction
                    expected = prior['auto_probs']
                    key = 'C_auto_vs_previous'
                if expected is not None:
                    error = float(np.max(np.abs(probs - expected)))
                    max_error[key] = max(max_error[key], error)
                    assert error < 1e-4, (sid, arm, kind, error)
            cv2.imwrite(str(OUT / 'images' / f'{sid}_{kind}.jpg'), roi)
        output_rows.append(row)
        overlay = image.copy()
        for box, color in [(row['manual_box'], (0,255,0)), (row['auto_box'], (0,165,255))]:
            if box is not None:
                x1,y1,x2,y2 = [int(round(x)) for x in box]
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 3)
        cv2.imwrite(str(OUT / 'images' / f'{sid}_boxes.jpg'), overlay)
        label = row['label'] if row['has_head'] else '无可见头部（不评估深度类别）'
        changed = row['B_auto_prediction'] != row['C_auto_prediction']
        missed = row['has_head'] and row['auto_box'] is None
        cards.append(f'<article data-diff="{int(changed)}" data-miss="{int(missed)}"><h3>{html.escape(sid)}</h3><p>真值：{label}；IoU={row["selected_iou"]:.3f}；检测置信度={row["confidence"]}</p><table><tr><th>模型</th><th>人工框</th><th>自动框</th></tr>'+''.join(f'<tr><td>{arm}</td><td>{row[arm+"_manual_prediction"]}</td><td>{row[arm+"_auto_prediction"]}</td></tr>' for arm in ARMS)+f'</table><img width="540" src="images/{sid}_boxes.jpg">'+''.join(f'<img width="192" src="images/{sid}_{kind}.jpg">' for kind in ['manual','auto'] if row[kind+'_box'] is not None)+'</article>')
    positive = [r for r in output_rows if r['has_head']]
    paired = [r for r in positive if r['auto_box'] is not None]
    summary = {'head_frames':len(positive), 'accepted_head_frames':len(paired), 'missed_frames':len(positive)-len(paired), 'baseline_max_probability_errors':max_error}
    for arm in ARMS:
        summary[arm] = {
            'manual_all': metrics(positive, f'{arm}_manual_prediction'),
            'automatic_all_including_misses': metrics(positive, f'{arm}_auto_prediction'),
            'manual_detected_subset': metrics(paired, f'{arm}_manual_prediction'),
            'automatic_detected_subset': metrics(paired, f'{arm}_auto_prediction'),
            'manual_correct_to_auto_wrong': [r['sample_id'] for r in positive if r[f'{arm}_manual_prediction']==r['label'] and r[f'{arm}_auto_prediction']!=r['label']],
            'manual_wrong_to_auto_correct': [r['sample_id'] for r in positive if r[f'{arm}_manual_prediction']!=r['label'] and r[f'{arm}_auto_prediction']==r['label']],
        }
    summary['B_C_auto_disagreements'] = [r['sample_id'] for r in positive if r['B_auto_prediction']!=r['C_auto_prediction']]
    summary['B_only_auto_correct'] = [r['sample_id'] for r in positive if r['B_auto_prediction']==r['label'] and r['C_auto_prediction']!=r['label']]
    summary['C_only_auto_correct'] = [r['sample_id'] for r in positive if r['C_auto_prediction']==r['label'] and r['B_auto_prediction']!=r['label']]
    dump(OUT/'comparison_predictions.json', output_rows)
    dump(OUT/'comparison_summary.json', summary)
    fields = ['sample_id','label','has_head','confidence','selected_iou','B_manual_prediction','B_auto_prediction','C_manual_prediction','C_auto_prediction','manual_box','auto_box']
    with (OUT/'comparison_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(output_rows)
    lines = ['# 固定 B 与 C：人工框 → 相同自动框', '', 'B = 头部裁剪像素，不输入显式尺度；C = 相同头部裁剪像素 + 未扩框的宽高/FOV直径。B并非“没有头部像素信息”。', '', '## 对照协议', '', '- 使用原有B/C seed42权重和相同冻结ConvNeXt backbone，不训练、不选择新权重。', '- 直接复用上一轮保存的FP32检测候选与最高置信度框（conf=0.25，NMS IoU=0.7）；不重跑检测、不调阈值、不使用真值补漏检。', '- 两组均扩框1.2倍、等比例填充384。B的两个辅助输入固定为0，与B训练一致；C使用原checkpoint的尺度归一化参数。', '- 36张验证图像中的34张有头部计入分类；4张漏检记Invalid并计入对应真值类别FN。另2张无头部不评估深度类别，其自动框仍为空。', '', '## 全部34张有头部验证图像', '', '|模型|输入框|正确/总数|准确率|Macro-F1|', '|---|---|---:|---:|---:|']
    for arm in ARMS:
        for key, name in [('manual_all','人工框'),('automatic_all_including_misses','自动框（含漏检）')]:
            m = summary[arm][key]
            correct = sum(m['confusion'][i][i] for i in range(3))
            lines.append(f'|{arm}|{name}|{correct}/{m["n"]}|{m["accuracy"]:.2%}|{m["macro_f1"]:.2%}|')
    lines += ['', '## 相同的30张检测成功子集', '', '|模型|人工框准确率|自动框准确率|', '|---|---:|---:|']
    for arm in ARMS:
        lines.append(f'|{arm}|{summary[arm]["manual_detected_subset"]["accuracy"]:.2%}|{summary[arm]["automatic_detected_subset"]["accuracy"]:.2%}|')
    lines += ['', '## 自动框分类混淆矩阵', '', '行：TooFar、Good、TooClose；列：TooFar、Good、TooClose、Invalid。']
    for arm in ARMS:
        lines += ['', arm, '```', str(np.array(summary[arm]['automatic_all_including_misses']['confusion'])), '```']
    lines += ['', '## B/C自动框分歧', '', '|样本|真值|B|C|', '|---|---|---|---|']
    for r in positive:
        if r['sample_id'] in summary['B_C_auto_disagreements']:
            lines.append(f'|{r["sample_id"]}|{r["label"]}|{r["B_auto_prediction"]}|{r["C_auto_prediction"]}|')
    if not summary['B_C_auto_disagreements']:
        lines.append('|无类别分歧|—|—|—|')
    lines += ['', '## 复核与解释边界', '', f'人工框B/C与原结果、自动框C与上一轮结果的概率最大差：`{max_error}`。输入图像、标注和所有冻结权重已核验；哈希见completion.json。', '', '同视频混合开发验证集，曾用于模型选择；样本仅34张，不是独立或真实手术泛化测试。B/C具有独立训练的分类头，这项对照比较两种已训练方案，不能将差异全部解释成“尺度特征”的单独因果效应。', '', '去掉显式尺度不会解决漏检；即使使用B，框的位置和尺寸变化仍会改变裁剪像素内容、填充比例。没有重新测试耗时，检测器与上一轮完全相同。', '', '复现：`python scripts/evaluate_head_replacement_bc.py`。输出目录已有completion.json时会停止，防止覆盖本次结果。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'comparison_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>固定 B/C 自动框对照</title><style>body{font-family:sans-serif;background:#eee;margin:24px}article{background:white;padding:16px;margin:16px 0}img{vertical-align:middle;margin:4px}td,th{padding:6px 20px;border:1px solid #ccc}table{border-collapse:collapse}button{margin:8px;padding:8px}</style><h1>固定 B/C：人工框 → 相同自动框</h1><p>B：裁剪像素，无显式尺度；C：裁剪像素＋尺度。绿色=人工框，橙色=自动框。裁剪依次为人工框与自动框。</p><button onclick="show(\'all\')">全部</button><button onclick="show(\'diff\')">B/C自动框类别不同</button><button onclick="show(\'miss\')">漏检</button><p>完整指标见同目录report.md；漏检计入34张分类分母。</p>'+''.join(cards)+'<script>function show(mode){document.querySelectorAll("article").forEach(e=>e.hidden=mode!=="all"&&e.dataset[mode]!=="1")}</script>', encoding='utf-8')
    for path, digest in hashes.items():
        assert sha(path) == digest, path
    dump(OUT/'completion.json', dict(frozen_file_hashes=hashes,script_sha256=sha(__file__),device=torch.cuda.get_device_name(),torch_version=torch.__version__,summary=summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
