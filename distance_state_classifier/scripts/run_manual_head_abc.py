#!/usr/bin/env python3
"""Prepare, train and report the frozen ConvNeXt manual-head A/B/C experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from distance_state_classifier.src.manual_head_abc import (
    FusionHead, make_backbone, parse_head_box, prepare_inputs,
    image_tensor, fit_scale_normalizer, expanded_box)
from distance_state_classifier_endodac.src.metrics import classification_metrics, confusion_matrix

ARMS = {'A_full': '原图', 'B_roi': '头部裁剪', 'C_roi_scale': '头部裁剪＋尺度'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_csv(path, records):
    if not records:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def source_fingerprints():
    paths = [Path(__file__), ROOT / 'distance_state_classifier/src/manual_head_abc.py',
             ROOT / 'distance_state_classifier/src/head_roi.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def prepare(cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'inputs').mkdir()
    package = ROOT / cfg['package']
    manifest = package / 'manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    calibration = json.loads((ROOT / cfg['fov_audit']).read_text())['calibration']
    hashes = {str(manifest): digest(manifest), str(ROOT / cfg['fov_audit']): digest(ROOT / cfg['fov_audit'])}
    records, excluded, cards, seen = [], [], [], set()
    for row in source:
        split = row['annotation_split']
        if split not in ('train', 'val'):
            raise ValueError('Unexpected current annotation_split')
        image_path = package / row['annotation_image']
        json_path = package / row['expected_annotation_json']
        ih, jh = digest(image_path), digest(json_path)
        if ih != row['image_sha256'] or ih in seen:
            raise ValueError(f'Changed or duplicate source image: {image_path}')
        seen.add(ih)
        hashes[str(image_path)], hashes[str(json_path)] = ih, jh
        annotation = json.loads(json_path.read_text())
        if annotation.get('imagePath') != image_path.name:
            raise ValueError(f'Annotation/image mismatch: {json_path}')
        if not annotation.get('shapes'):
            excluded.append({'sample_id': row['sample_id'], 'split': split, 'label': row['distance_label'],
                             'reason': 'empty_annotation_no_head_box'})
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(str(image_path))
        height, width = image.shape[:2]
        if [annotation['imageHeight'], annotation['imageWidth']] != [height, width]:
            raise ValueError('Annotation dimensions mismatch')
        box = parse_head_box(annotation, width, height)
        fov = calibration[row['video_id']]
        if fov['shape'] != [height, width]:
            raise ValueError('FOV/image coordinates differ')
        full, local, scales = prepare_inputs(image, box, fov['diameter'], cfg['input_size'], cfg['context_factor'])
        paths = {kind: f'inputs/{row["sample_id"]}_{kind}.png' for kind in ('full', 'roi')}
        for kind, array in (('full', full), ('roi', local)):
            if not cv2.imwrite(str(output / paths[kind]), array):
                raise RuntimeError('Failed to save prepared input')
        border = box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1
        record = {'sample_id': row['sample_id'], 'split': split, 'label': row['distance_label'],
                  'target': cfg['classes'].index(row['distance_label']), 'video_id': row['video_id'],
                  'image_path': str(image_path), 'annotation_path': str(json_path), 'box': box,
                  'expanded_box': expanded_box(box, cfg['context_factor']), 'fov_diameter': fov['diameter'],
                  'scale_raw': scales.tolist(), 'touches_image_border': bool(border),
                  'full_input': paths['full'], 'roi_input': paths['roi'],
                  'full_sha256': digest(output / paths['full']), 'roi_sha256': digest(output / paths['roi'])}
        records.append(record)
        cards.append(f'<article><h3>{html.escape(row["sample_id"])} · {split}</h3>'
                     f'<p>原始框 w/D={scales[0]:.4f}, h/D={scales[1]:.4f}；触边={border}</p>'
                     f'<img loading="lazy" src="{paths["full"]}"><img loading="lazy" src="{paths["roi"]}"></article>')
    counts = {s: dict(Counter(r['label'] for r in records if r['split'] == s)) for s in ('train', 'val')}
    if [sum(counts[s].values()) for s in ('train', 'val')] != [94, 34]:
        raise ValueError(f'Dataset changed relative to the agreed 94/34 protocol: {counts}')
    train_mask = [r['split'] == 'train' for r in records]
    mean, std = fit_scale_normalizer([r['scale_raw'] for r in records], train_mask)
    audit = {'counts': counts, 'included_n': len(records), 'excluded_empty_n': len(excluded),
             'scale_mean_train': mean.tolist(), 'scale_std_train': std.tolist(),
             'source_hashes': hashes, 'code_hashes': source_fingerprints(),
             'pretrained_sha256': digest(Path(cfg['pretrained_file'])),
             'border_samples': [r['sample_id'] for r in records if r['touches_image_border']],
             'split_field': 'annotation_split; original_split never used for training',
             'evaluation': 'Mixed-video validation selected checkpoints; not an independent test',
             'empty_annotations': 'Excluded identically from A/B/C, even though A could accept full images'}
    save_json(output / 'config.json', cfg)
    save_json(output / 'prepared.json', {'records': records, 'audit': audit})
    save_json(output / 'data_audit.json', audit)
    save_csv(output / 'excluded_samples.csv', excluded)
    page = ('<!doctype html><meta charset="utf-8"><title>A/B/C 实际输入</title>'
            '<style>body{font-family:sans-serif;max-width:1000px;margin:auto}article{border-bottom:1px solid #ccc;padding:12px}img{width:384px;max-width:48%}</style>'
            '<h1>A/B/C 实际输入预览</h1><p>左：A 原图等比例填充；右：B/C 共用的头部裁剪。C 另接收裁剪前框宽高／视野直径。黑边为填充，无图像拉伸。此页不显示类别，便于检查裁剪。</p>'
            + ''.join(cards))
    (output / 'input_preview.html').write_text(page, encoding='utf-8')
    print('[prepared]', json.dumps(counts), 'excluded', len(excluded), flush=True)


def verify_sources(output, data):
    for name, expected in {**data['audit']['source_hashes'], **data['audit']['code_hashes']}.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if digest(path) != expected:
            raise ValueError(f'Source changed after preparation: {path}')
    for row in data['records']:
        for kind in ('full', 'roi'):
            if digest(output / row[f'{kind}_input']) != row[f'{kind}_sha256']:
                raise ValueError('Prepared input changed')


def metrics(targets, logits, classes):
    loss = nn.functional.cross_entropy(logits, targets).item()
    pred = logits.argmax(-1).cpu().tolist()
    truth = targets.cpu().tolist()
    cm = confusion_matrix(truth, pred, len(classes))
    result = classification_metrics(cm, classes)
    result.update(loss=loss, confusion_matrix=cm.tolist(), n=len(truth),
                  opposite_errors=sum({a, b} == {0, 2} for a, b in zip(truth, pred)))
    return result


def train(cfg, output, device):
    if (output / 'completion.json').exists() or (output / 'runs').exists():
        raise FileExistsError('Training outputs already exist; use a new output/config for a new experiment')
    data = json.loads((output / 'prepared.json').read_text())
    verify_sources(output, data)
    if digest(Path(cfg['pretrained_file'])) != data['audit']['pretrained_sha256']:
        raise ValueError('Pretraining file changed')
    records = data['records']
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = make_backbone(cfg['pretrained_file']).to(device)
    frozen_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    torch.save({'model_state': frozen_state, 'architecture': 'convnext_tiny; reset_classifier(0)',
                'pretrained_sha256': data['audit']['pretrained_sha256']}, output / 'frozen_backbone.pt')
    del frozen_state
    feature_sets = {}
    started = time.monotonic()
    for kind in ('full', 'roi'):
        chunks = []
        for start in range(0, len(records), cfg['feature_batch_size']):
            selected = records[start:start + cfg['feature_batch_size']]
            images = torch.stack([image_tensor(cv2.imread(str(output / r[f'{kind}_input']))) for r in selected]).to(device)
            with torch.inference_mode():
                chunks.append(backbone(images).cpu())
            if start % 32 == 0:
                print(f'[features {kind}] {start + len(selected)}/{len(records)}', flush=True)
        feature_sets[kind] = torch.cat(chunks)
        assert feature_sets[kind].shape == (len(records), 768)
        assert torch.isfinite(feature_sets[kind]).all()
    torch.save({'sample_ids': [r['sample_id'] for r in records], **feature_sets}, output / 'frozen_features.pt')
    feature_seconds = time.monotonic() - started
    train_indices = torch.tensor([i for i, r in enumerate(records) if r['split'] == 'train'], device=device)
    val_indices = torch.tensor([i for i, r in enumerate(records) if r['split'] == 'val'], device=device)
    target = torch.tensor([r['target'] for r in records], device=device)
    raw = torch.tensor([r['scale_raw'] for r in records], device=device)
    mean = torch.tensor(data['audit']['scale_mean_train'], device=device)
    std = torch.tensor(data['audit']['scale_std_train'], device=device)
    normalized = (raw - mean) / std
    all_results, prediction_rows, primary_heads = [], [], {}
    for seed in cfg['seeds']:
        for arm in ARMS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
            head = FusionHead(cfg['hidden'], cfg['dropout']).to(device)
            initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            initial_hash = hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
            visual = feature_sets['full' if arm == 'A_full' else 'roi'].to(device)
            scales = normalized if arm == 'C_roi_scale' else torch.zeros_like(normalized)
            optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            generator = torch.Generator().manual_seed(seed)
            best, stale, history = -1., 0, []
            run_dir = output / 'runs' / arm / f'seed_{seed}'
            run_dir.mkdir(parents=True, exist_ok=False)
            begin = time.monotonic()
            for epoch in range(1, cfg['epochs'] + 1):
                head.train()
                order = torch.randperm(len(train_indices), generator=generator).to(device)
                for start in range(0, len(order), cfg['batch_size']):
                    idx = train_indices[order[start:start + cfg['batch_size']]]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(head(visual[idx], scales[idx]), target[idx])
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradients')
                    optimizer.step()
                head.eval()
                with torch.inference_mode():
                    train_metric = metrics(target[train_indices], head(visual[train_indices], scales[train_indices]), cfg['classes'])
                    val_metric = metrics(target[val_indices], head(visual[val_indices], scales[val_indices]), cfg['classes'])
                history.append({'epoch': epoch, 'train': train_metric, 'val': val_metric})
                if val_metric['macro_f1'] > best + 1e-12:
                    best, stale = val_metric['macro_f1'], 0
                    best_payload = {'model_state': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                                    'config': cfg, 'arm': arm, 'seed': seed, 'epoch': epoch,
                                    'scale_mean': mean.cpu(), 'scale_std': std.cpu(),
                                    'backbone_file': str(output / 'frozen_backbone.pt'),
                                    'validation': val_metric, 'training': train_metric,
                                    'initial_head_sha256': initial_hash,
                                    'prepared_sha256': digest(output / 'prepared.json')}
                    torch.save(best_payload, run_dir / 'best.pt')
                else:
                    stale += 1
                if stale >= cfg['patience']: break
            head.load_state_dict(best_payload['model_state'], strict=True)
            head.eval()
            with torch.inference_mode():
                all_probs = head(visual, scales).softmax(-1).cpu().numpy()
            for i, row in enumerate(records):
                probs = all_probs[i]
                prediction_rows.append({'arm': arm, 'seed': seed, 'sample_id': row['sample_id'], 'split': row['split'],
                                        'video_id': row['video_id'], 'label': row['label'],
                                        'prediction': cfg['classes'][int(probs.argmax())],
                                        **{f'p_{label}': float(p) for label, p in zip(cfg['classes'], probs)}})
            result = {'arm': arm, 'seed': seed, 'best_epoch': best_payload['epoch'], 'last_epoch': epoch,
                      'train': best_payload['training'], 'val': best_payload['validation'],
                      'initial_head_sha256': initial_hash, 'seconds': time.monotonic() - begin}
            save_json(run_dir / 'history.json', history)
            save_json(run_dir / 'result.json', result)
            all_results.append(result)
            if seed == cfg['primary_seed']:
                primary_heads[arm] = head
            print(f'[complete {arm} seed={seed}] best={result["best_epoch"]} stop={epoch} val_acc={result["val"]["accuracy"]:.4f} val_f1={best:.4f}', flush=True)
    for seed in cfg['seeds']:
        assert len({r['initial_head_sha256'] for r in all_results if r['seed'] == seed}) == 1
    # Roundtrip shared backbone checkpoint and raw-image preprocessing on one
    # validation image per class. B/C must use identical visual features.
    backbone.load_state_dict(torch.load(output / 'frozen_backbone.pt', map_location='cpu', weights_only=True)['model_state'], strict=True)
    checks, max_error = [], 0.
    for label in cfg['classes']:
        index = next(i for i, r in enumerate(records) if r['split'] == 'val' and r['label'] == label)
        row = records[index]
        full, roi, scale = prepare_inputs(cv2.imread(row['image_path']), row['box'], row['fov_diameter'], cfg['input_size'], cfg['context_factor'])
        for kind, array in (('full', full), ('roi', roi)):
            if not np.array_equal(array, cv2.imread(str(output / row[f'{kind}_input']))):
                raise AssertionError('Live input differs from PNG cache')
            with torch.inference_mode():
                live = backbone(image_tensor(array)[None].to(device))
            torch.testing.assert_close(live.cpu(), feature_sets[kind][index:index + 1], rtol=1e-4, atol=1e-4)
            for arm in (['A_full'] if kind == 'full' else ['B_roi', 'C_roi_scale']):
                restored = FusionHead(cfg['hidden'], cfg['dropout']).to(device).eval()
                saved = torch.load(output / 'runs' / arm / f'seed_{cfg["primary_seed"]}' / 'best.pt', map_location='cpu', weights_only=True)
                restored.load_state_dict(saved['model_state'], strict=True)
                s = ((torch.tensor(scale, device=device) - mean) / std)[None] if arm == 'C_roi_scale' else torch.zeros(1, 2, device=device)
                with torch.inference_mode():
                    live_probs = restored(live, s).softmax(-1).cpu().numpy()[0]
                cached = next(r for r in prediction_rows if r['arm'] == arm and r['seed'] == cfg['primary_seed'] and r['sample_id'] == row['sample_id'])
                expected = np.array([cached[f'p_{c}'] for c in cfg['classes']])
                np.testing.assert_allclose(live_probs, expected, rtol=1e-4, atol=1e-4)
                max_error = max(max_error, float(np.abs(live_probs - expected).max()))
                checks.append({'arm': arm, 'sample_id': row['sample_id']})
    verify_sources(output, data)
    save_csv(output / 'predictions.csv', prediction_rows)
    save_json(output / 'results.json', all_results)
    save_json(output / 'inference_checks.json', {'cases': checks, 'max_probability_error': max_error,
                                               'shared_initialization_per_seed': True,
                                               'trainable_parameters': sum(p.numel() for p in FusionHead().parameters())})
    save_json(output / 'completion.json', {'status': 'complete', 'device': str(device),
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'torch': str(torch.__version__), 'feature_seconds': feature_seconds, 'runs': len(all_results),
              'backbone_sha256': digest(output / 'frozen_backbone.pt'), 'source_hashes': source_fingerprints()})


def report(cfg, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = json.loads((output / 'results.json').read_text())
    complete = json.loads((output / 'completion.json').read_text())
    data = json.loads((output / 'prepared.json').read_text())
    with (output / 'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        predictions = list(csv.DictReader(f))
    aggregate = {}
    for arm in ARMS:
        selected = [r for r in results if r['arm'] == arm]
        aggregate[arm] = {metric: {'mean': float(np.mean([r['val'][metric] for r in selected])),
                                  'std': float(np.std([r['val'][metric] for r in selected], ddof=1))}
                          for metric in ('accuracy', 'macro_f1', 'balanced_accuracy')}
        aggregate[arm]['recall'] = [{'mean': float(np.mean([r['val']['per_class'][i]['recall'] for r in selected])),
                                    'std': float(np.std([r['val']['per_class'][i]['recall'] for r in selected], ddof=1))} for i in range(3)]
    save_json(output / 'aggregate.json', aggregate)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.arange(3)
    for i, key in enumerate(('accuracy', 'macro_f1')):
        axes[0].bar(x + (i-.5)*.35, [aggregate[a][key]['mean'] for a in ARMS], .35,
                    yerr=[aggregate[a][key]['std'] for a in ARMS], capsize=3, label=key)
    axes[0].set(xticks=x, xticklabels=['A full', 'B ROI', 'C ROI+scale'], ylim=(0, 1.1), title='Validation: 3 seeds, mean +/- SD')
    axes[0].legend()
    for i, arm in enumerate(ARMS):
        history = json.loads((output / 'runs' / arm / f'seed_{cfg["primary_seed"]}' / 'history.json').read_text())
        for split, style in (('train', '--'), ('val', '-')):
            axes[1].plot([h['epoch'] for h in history], [h[split]['macro_f1'] for h in history], style, color=f'C{i}', label=f'{arm[0]} {split}')
        axes[2].bar(x+(i-1)*.25, [r['mean'] for r in aggregate[arm]['recall']], .25, label=arm[0])
    axes[1].set(xlabel='Epoch', ylabel='Macro-F1', ylim=(0, 1.05), title='Seed 42 training / validation'); axes[1].legend(fontsize=8)
    axes[2].set(xticks=x, xticklabels=cfg['classes'], ylim=(0, 1.1), title='Validation recall: mean over seeds'); axes[2].legend()
    fig.savefig(output / 'overview.png', dpi=180); plt.close(fig)
    fmt = lambda obj: f'{100*obj["mean"]:.2f} ± {100*obj["std"]:.2f}'
    lines = ['# ConvNeXt 人工头部框 A/B/C 对照实验', '',
             f'三组实验各完成 3 个种子（42、43、44），共 {complete["runs"]} 次分类头训练。设备：{complete["gpu"] or complete["device"]}。', '',
             '## 数据与公平性', '',
             '- 相同的 94 张训练图、34 张验证图。训练 TooFar/Good/TooClose=31/30/33；验证=5/15/14。',
             '- 空标注训练 50 张、验证 2 张从三组共同排除；结果仅针对可提供人工头部框的样本。',
             '- 使用重新混合后的 annotation_split，训练和验证有相同视频来源；没有独立测试集结果。',
             '- 三组从同一本地 ImageNet 预训练 ConvNeXt-Tiny 开始，未加载旧距离分类权重。',
             '- 主干全程冻结并处于 eval 模式；缓存确定性特征等价于每步运行冻结主干，不使用数据增强。',
             '- 三组分类头结构、参数数量、同种子初始化及批次顺序一致。A/B 的两个尺度通道恒为零，C 使用训练集标准化后的宽高／FOV。',
             '- B/C 使用完全相同的 ROI 和视觉特征缓存；裁剪框宽高各扩大 1.2 倍，越界补零，等比例缩放并填充到 384×384。A 同样等比例填充。',
             '- 普通三类交叉熵，无类别权重；AdamW，LR=3e-4，weight_decay=1e-3，batch=32，最多100轮，patience=20。',
             '- 每次按验证集 macro-F1 选择最佳轮次，同分保留更早轮次；验证集参与了模型选择，表格不是无偏的独立测试性能。', '',
             '## 三个种子的均值 ± 样本标准差（百分点）', '',
             '| 方案 | Accuracy | Macro-F1 | TooFar recall | Good recall | TooClose recall |', '|---|---:|---:|---:|---:|---:|']
    for arm, name in ARMS.items():
        a = aggregate[arm]
        lines.append(f'| {arm}: {name} | {fmt(a["accuracy"])} | {fmt(a["macro_f1"])} | '+ ' | '.join(fmt(r) for r in a['recall'])+' |')
    lines += ['', '![对照与训练曲线](overview.png)', '', '## 每次训练', '',
              '| 方案 | Seed | 最佳轮次 | 停止轮次 | Train accuracy | Val accuracy | Val macro-F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["best_epoch"]} | {r["last_epoch"]} | {100*r["train"]["accuracy"]:.2f}% | {100*r["val"]["accuracy"]:.2f}% | {100*r["val"]["macro_f1"]:.2f}% |')
    lines += ['', '## 预先指定 Seed 42 的混淆矩阵', '', '行是真实类别、列为预测类别，顺序 TooFar、Good、TooClose。这里没有选择表现最好的种子。', '']
    for r in results:
        if r['seed'] == cfg['primary_seed']:
            lines += [r['arm'], '', '```text', *[str(row) for row in r['val']['confusion_matrix']], '```', '']
    lines += ['## 解释边界', '',
              '- 三个种子的标准差只反映分类头初始化及优化的波动，不是跨病例置信区间。验证 TooFar 只有5张，每张对应召回率20个百分点。',
              '- 人工框提供理想定位条件，不能推断自动检测后的端到端性能或真实手术泛化。',
              '- 相同器械的姿态、张合、遮挡可能改变框宽高；宽高／FOV 不是毫米级距离。',
              '- 触及原图边界的标注保留在三组共同样本中，没有根据验证效果人工删除；触边列表见 data_audit.json。',
              '- 本轮使用全局池化的冻结 ConvNeXt 特征；没有启用有序头、像素 mask、额外空间分区或历史实验的增强。',
              '- 保存的 best.pt 是分类头权重，需配合共享 frozen_backbone.pt 和其预处理／尺度标准化信息使用。', '',
              '## 文件', '', '[实际模型输入预览](input_preview.html) · [逐图三组预测](prediction_gallery.html) · [预测 CSV](predictions.csv) · [原始统计](aggregate.json)', '',
              '各组每个种子的权重：`runs/{A_full,B_roi,C_roi_scale}/seed_{42,43,44}/best.pt`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    cards = []
    for row in data['records']:
        if row['split'] != 'val': continue
        selected = [p for p in predictions if p['sample_id'] == row['sample_id'] and int(p['seed']) == cfg['primary_seed']]
        details = '<br>'.join(f'{p["arm"]}: {p["prediction"]}'+(' ✓' if p['prediction']==row['label'] else ' ✗') for p in selected)
        cards.append(f'<article><h3>{row["sample_id"]} · 标签 {row["label"]}</h3><p>{details}</p>'
                     f'<img loading="lazy" src="{row["full_input"]}"><img loading="lazy" src="{row["roi_input"]}"></article>')
    (output / 'prediction_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>A/B/C 验证预测</title><style>body{font-family:sans-serif;max-width:1000px;margin:auto}article{padding:12px;border-bottom:1px solid #aaa}img{width:384px;max-width:48%}</style><h1>验证集三组预测（固定 Seed 42）</h1>'+''.join(cards), encoding='utf-8')
    print('[report]', output / 'report.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='distance_state_classifier/configs/manual_head_abc.json')
    parser.add_argument('--stage', choices=['prepare', 'train', 'report', 'all'], default='all')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    cfg = json.loads((ROOT / args.config).read_text())
    output = ROOT / cfg['output']
    if args.stage in ('prepare', 'all'): prepare(cfg, output)
    else:
        if json.loads((output / 'config.json').read_text()) != cfg:
            raise ValueError('Configuration differs from prepared protocol')
    if args.stage in ('train', 'all'):
        lock = output / '.training.lock'
        with lock.open('x') as handle: handle.write(str(time.time()))
        try: train(cfg, output, torch.device(args.device))
        finally: lock.unlink()
    if args.stage in ('report', 'all'): report(cfg, output)


if __name__ == '__main__':
    main()
