#!/usr/bin/env python3
"""Audit manual head boxes and render an independent, non-mutating review page."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='annotation_projects/forceps_head_pilot_20260907')
    parser.add_argument('--batch', default='01_pilot_train')
    parser.add_argument('--output', default='pilot_review_20260917')
    args = parser.parse_args()
    package = ROOT / args.package
    output = package / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    with (package / 'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        samples = [r for r in csv.DictReader(handle) if r['annotation_batch'] == args.batch]
    prior_path = ROOT / 'results/convnext_head_roi_cache_20260907/manifest.json'
    prior = {r['sample_id']: r for r in json.loads(prior_path.read_text())['records']} if prior_path.exists() else {}
    records, cards = [], []
    for row in samples:
        sample_id = row['sample_id']
        path = package / row['annotation_image']
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Unreadable image: {path}')
        height, width = image.shape[:2]
        issues = []
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            issues.append('image_changed_since_preparation')
        label_path = path.with_suffix('.json')
        shapes = []
        status = 'missing_json'
        if label_path.exists():
            data = json.loads(label_path.read_text(encoding='utf-8'))
            shapes = data.get('shapes', [])
            status = 'has_box' if shapes else 'empty_reason_unconfirmed'
            if (data.get('imageWidth'), data.get('imageHeight')) != (width, height):
                issues.append('image_dimensions_mismatch')
            if Path(data.get('imagePath', '')).name != path.name:
                issues.append('image_path_mismatch')
        boxes = []
        for shape in shapes:
            points = np.asarray(shape.get('points', []), dtype=float)
            if shape.get('label') != 'forceps_head' or shape.get('shape_type') != 'rectangle':
                issues.append('unexpected_label_or_shape')
                continue
            # X-AnyLabeling can store a rectangle as two corners or four vertices.
            if points.shape not in ((2, 2), (4, 2)) or not np.isfinite(points).all():
                issues.append('invalid_points')
                continue
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            if x2 <= x1 or y2 <= y1:
                issues.append('nonpositive_box')
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issues.append('box_outside_image')
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        if len(boxes) > 1:
            issues.append('multiple_heads_require_target_selection')
        border = any(x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1 for x1, y1, x2, y2 in boxes)
        geometry = prior.get(sample_id, {}).get('geometry', {})
        coverage = None
        if len(boxes) == 1 and geometry.get('valid') and 'box' in geometry:
            x1, y1, x2, y2 = boxes[0]
            a, b, c, d = geometry['box']
            coverage = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b)) / ((x2 - x1) * (y2 - y1))
        annotated = image.copy()
        if 'box' in geometry:
            a, b, c, d = map(round, geometry['box'])
            cv2.rectangle(annotated, (a, b), (c, d), (0, 170, 255), 3)
        for box in boxes:
            x1, y1, x2, y2 = map(round, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 220, 0), 3)
        relative = f'images/{sample_id}.jpg'
        if not cv2.imwrite(str(output / relative), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError('Preview save failed')
        details = f'状态：{status}；人工框数：{len(boxes)}'
        if coverage is not None:
            details += f'；旧 ROI 对人工框的面积保留比例：{coverage:.1%}'
        if border:
            details += '；需人工复核：头部框触及原图边界'
        if issues:
            details += '；格式问题：' + ', '.join(issues)
        cards.append(f'<article id="{html.escape(sample_id)}"><h2>{html.escape(sample_id)}</h2>'
                     f'<p>{html.escape(details)}</p><a href="{relative}"><img loading="lazy" src="{relative}"></a></article>')
        records.append({'sample_id': sample_id, 'distance_label': row['distance_label'],
                        'status': status, 'box_count': len(boxes), 'boxes_xyxy': json.dumps(boxes),
                        'touches_image_border': border, 'old_roi_valid': geometry.get('valid', False),
                        'old_roi_box_retention': coverage, 'format_issues': ';'.join(issues),
                        'annotation_json': str(label_path.relative_to(package))})
    counts = Counter(r['status'] for r in records)
    per_class = {label: dict(Counter(r['status'] for r in records if r['distance_label'] == label))
                 for label in ('TooFar', 'Good', 'TooClose')}
    covered = [r['old_roi_box_retention'] for r in records if r['old_roi_box_retention'] is not None]
    summary = {'total': len(records), 'status_counts': dict(counts), 'by_distance_class': per_class,
               'format_issue_samples': [r['sample_id'] for r in records if r['format_issues']],
               'border_touching_samples': [r['sample_id'] for r in records if r['touches_image_border']],
               'old_roi_comparison_n': len(covered),
               'old_roi_retention_at_least_095': sum(v >= .95 for v in covered),
               'old_roi_retention_below_050': sum(v < .5 for v in covered),
               'old_roi_mean_box_retention': float(np.mean(covered)) if covered else None,
               'note': 'Box retention measures rectangle overlap, not true head-pixel coverage. Manual annotation semantics and empty-image reasons are not validated.'}
    with (output / 'audit.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人工头部框审查</title>'
            '<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#f7f8fa}'
            'article{background:white;padding:16px;margin:18px 0;border:1px solid #ddd}img{max-width:100%;width:850px}'
            'p{line-height:1.7}h2{font-size:18px}</style><h1>第一批人工头部框审查</h1>'
            '<p><b>青色：人工头部框；橙色：旧固定 ROI。</b>点击图片查看原尺寸叠框图。原图和 JSON 均未修改。</p>'
            '<p>空标注的原因尚未确认，不自动认定为负样本。检查：框的根部定义是否一致、是否包含两个钳瓣、是否把长杆或夹持物计入头部、是否被原图边界截断。</p>'
            '<p>面积保留比例是“人工矩形框与旧 ROI 的交集面积／人工框面积”，不是头部像素覆盖率。95% 只是诊断统计阈值，不代表定位合格标准；标注正确性仍需人工判断。</p>'
            + ''.join(cards) + '</html>')
    (output / 'gallery.html').write_text(page, encoding='utf-8')
    lines = ['# 第一批头部框审查', '', f'共 {len(records)} 张。此检查不修改人工标注或标注状态表。', '',
             '| 距离类别 | 有框 | 空标注 |', '|---|---:|---:|']
    for label, stat in per_class.items():
        lines.append(f'| {label} | {stat.get("has_box", 0)} | {stat.get("empty_reason_unconfirmed", 0)} |')
    lines += ['', f'格式问题样本数：{len(summary["format_issue_samples"])}。格式通过不等于头部范围标注正确。',
              f'触及原图边界需人工复核：{", ".join(summary["border_touching_samples"]) or "无"}。', '',
              f'旧 ROI 与人工框可比较 {len(covered)} 张，其中人工框面积保留 ≥95% 的有 {summary["old_roi_retention_at_least_095"]} 张，低于 50% 的有 {summary["old_roi_retention_below_050"]} 张。',
              '这里比较的是矩形框面积，不是真实头部像素覆盖率，不能据此单独归因分类性能退化。', '',
              '空标注需要区分头部确实不在画面内、遮挡或截断而无法定框等原因。前者可作为头部检测的候选负样本，后者通常需要忽略或专门的困难样本规则。所有空标注均无可用头部框，不能直接进入当前人工框裁剪＋尺度分支实验。', '',
              '[查看叠框预览](gallery.html) · [逐帧审计](audit.csv)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output / 'gallery.html')


if __name__ == '__main__':
    main()
