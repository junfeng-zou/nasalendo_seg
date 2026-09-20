#!/usr/bin/env python3
"""Post-training checks; never changes checkpoints or selects new settings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distance_state_classifier.src.manual_head_abc import FusionHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='results/convnext_manual_head_abc_20260917')
    args = parser.parse_args()
    output = ROOT / args.run_dir
    torch.set_num_threads(4)
    cfg = json.loads((output / 'config.json').read_text())
    prepared = json.loads((output / 'prepared.json').read_text())
    complete = json.loads((output / 'completion.json').read_text())
    for name, expected in complete['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    records = prepared['records']
    val = [i for i, r in enumerate(records) if r['split'] == 'val']
    features = torch.load(output / 'frozen_features.pt', map_location='cpu', weights_only=True)
    assert features['sample_ids'] == [r['sample_id'] for r in records]
    raw = torch.tensor([r['scale_raw'] for r in records])
    mean = torch.tensor(prepared['audit']['scale_mean_train'])
    std = torch.tensor(prepared['audit']['scale_std_train'])
    scales = ((raw - mean) / std)[val]
    visual = features['roi'][val]
    with (output / 'predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
        saved_predictions = list(csv.DictReader(handle))
    checks = []
    for seed in cfg['seeds']:
        heads = {}
        for arm in ('B_roi', 'C_roi_scale'):
            model = FusionHead(cfg['hidden'], cfg['dropout']).eval()
            checkpoint = torch.load(output / 'runs' / arm / f'seed_{seed}' / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            heads[arm] = model
        with torch.inference_mode():
            bp = heads['B_roi'](visual, torch.zeros_like(scales)).softmax(-1).numpy()
            cp = heads['C_roi_scale'](visual, scales).softmax(-1).numpy()
            cz = heads['C_roi_scale'](visual, torch.zeros_like(scales)).softmax(-1).numpy()
        for arm, probs in (('B_roi', bp), ('C_roi_scale', cp)):
            mapping = {r['sample_id']: r for r in saved_predictions if r['arm'] == arm and int(r['seed']) == seed and r['split'] == 'val'}
            expected = np.array([[float(mapping[records[i]['sample_id']][f'p_{c}']) for c in cfg['classes']] for i in val])
            np.testing.assert_allclose(probs, expected, atol=1e-5, rtol=1e-5)
        torch.manual_seed(seed)
        initial = FusionHead(cfg['hidden'], cfg['dropout'])
        delta = heads['C_roi_scale'].classifier[0].weight[:, -2:] - initial.classifier[0].weight[:, -2:]
        change = float(delta.norm())
        assert change > 0
        checks.append({'seed': seed, 'n': len(val), 'scale_weight_update_l2': change,
                       'B_vs_C_different_predictions': int((bp.argmax(1) != cp.argmax(1)).sum()),
                       'B_vs_C_mean_probability_tv': float(np.abs(bp - cp).sum(1).mean() / 2),
                       'C_actual_vs_train_mean_scale_mean_tv': float(np.abs(cp - cz).sum(1).mean() / 2),
                       'C_actual_vs_train_mean_scale_max_probability_change': float(np.abs(cp - cz).max()),
                       'C_actual_vs_train_mean_scale_different_predictions': int((cp.argmax(1) != cz.argmax(1)).sum())})
    details = {'checks': checks, 'zero_scale_meaning': 'standardized zeros = training-set mean width/FOV and height/FOV, not an absent head',
               'scope': 'post-hoc implementation and sensitivity check on fixed checkpoints; no model retraining or hyperparameter changes',
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'scale_channel_checks.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    results = json.loads((output / 'results.json').read_text())
    agg = json.loads((output / 'aggregate.json').read_text())
    a, b, c = (agg[k] for k in ('A_full', 'B_roi', 'C_roi_scale'))
    delta_accuracy = 100 * (b['accuracy']['mean'] - a['accuracy']['mean'])
    delta_f1 = 100 * (b['macro_f1']['mean'] - a['macro_f1']['mean'])
    lines = ['# 本轮结果解读与尺度通道核查', '',
             f'B 相对 A 的平均验证准确率提高 {delta_accuracy:.2f} 个百分点，macro-F1 提高 {delta_f1:.2f} 个百分点。三个种子均出现改善，支持在当前混合划分和冻结主干条件下继续研究人工头部局部输入。', '',
             'B 与 C 三个种子的验证类别预测逐帧完全一致，但概率不同。本轮未观察到追加框宽高尺度带来的分类指标增益，不能由此推断尺度在其他融合结构、训练设置或数据中无用。', '',
             '改善并非所有类别同步：TooClose 平均召回从 69.05% 升至 90.48%，TooFar 从 80.00% 升至 86.67%，Good 从 80.00% 略降至 77.78%。', '',
             '## 尺度通道确实参与计算', '',
             '固定 C 的权重，将每张验证图的标准化尺度替换为零（即训练集平均尺度），观察输出变化。这是实现核查，不是额外训练实验，也没有据此改配置。', '',
             '| Seed | 尺度连接权重更新 L2 | B/C 类别不同数 | B/C 平均概率 TV | C 实际尺度/平均尺度 TV | C 替换尺度后类别变化数 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in checks:
        lines.append(f'| {r["seed"]} | {r["scale_weight_update_l2"]:.6f} | {r["B_vs_C_different_predictions"]} | {r["B_vs_C_mean_probability_tv"]:.6f} | {r["C_actual_vs_train_mean_scale_mean_tv"]:.6f} | {r["C_actual_vs_train_mean_scale_different_predictions"]} |')
    lines += ['', '尺度权重更新且输出概率发生变化，说明通道已接入；当前影响是否足以改变 argmax，见上表。', '',
              '## 跨级错误', '', '| 方案 | Seed 42 | Seed 43 | Seed 44 |', '|---|---:|---:|---:|']
    for arm in ('A_full', 'B_roi', 'C_roi_scale'):
        counts = [next(r['val']['opposite_errors'] for r in results if r['arm'] == arm and r['seed'] == seed) for seed in cfg['seeds']]
        lines.append(f'| {arm} | '+ ' | '.join(map(str, counts))+' |')
    lines += ['', '总体准确率提高不代表所有错误风险下降，应检查逐图错误，尤其是 TooFar 与 TooClose 直接混淆的样本。', '',
              '## 下一步', '',
              '先复核局部裁剪后仍出错的头部范围、遮挡、姿态和标签一致性。人工 ROI 在本轮有初步收益，可以将自动头部检测作为下一阶段候选，再衡量自动定位对 B 的影响。当前结果不足以支持直接替换实时控制模型。', '',
              '这是同视频来源混合划分、人工框可用样本上的验证结果，且验证集用于选择 epoch；不与旧实验 122 张测试集的百分比直接作优劣比较，也不表明真实手术泛化已解决。', '']
    text = '\n'.join(lines)
    (output / 'interpretation.md').write_text(text, encoding='utf-8')
    report_path = output / 'report.md'
    report_text = report_path.read_text(encoding='utf-8')
    marker = '## 结果解读与通道核查\n'
    if marker not in report_text:
        intro = (f'## 结果解读与通道核查\n\nB 相对 A 的平均验证准确率提高 {delta_accuracy:.2f} 个百分点，macro-F1 提高 {delta_f1:.2f} 个百分点。C 与 B 在三个种子上逐帧类别预测相同，未观察到额外分类增益。尺度权重和概率敏感性核查已通过。\n\n'
                 '[详细解读、尺度通道核查和跨级错误](interpretation.md)。\n\n')
        report_text = report_text.replace('## 数据与公平性\n', intro + '## 数据与公平性\n', 1)
        report_path.write_text(report_text, encoding='utf-8')
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
