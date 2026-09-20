#!/usr/bin/env python3
"""Exchange annotated image/JSON pairs, with provenance and rollback on failure."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ('01_pilot_train', '02_more_train', '03_validation')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8-sig')


def atomic_write(path, content):
    temporary = path.with_name(path.name + '.remix-tmp')
    with temporary.open('xb') as handle:
        handle.write(content)
    temporary.replace(path)


def inventory(package, rows):
    result = {}
    for row in rows:
        relative = Path(row['annotation_image'])
        path = package / relative
        if not path.is_file() or sha(path) != row['image_sha256']:
            raise ValueError(f'Missing or changed image: {path}')
        label = path.with_suffix('.json')
        if label.exists():
            data = json.loads(label.read_text(encoding='utf-8'))
            if data.get('imagePath') != path.name:
                raise ValueError(f'Expected a same-directory imagePath before moving: {label}')
            if not isinstance(data.get('shapes'), list):
                raise ValueError(f'Invalid shapes field: {label}')
        result[row['sample_id']] = {'image_sha256': sha(path),
                                    'annotation_sha256': sha(label) if label.exists() else None}
    if len(result) != len(rows):
        raise ValueError('Duplicate sample IDs')
    expected_images = {row['annotation_image'] for row in rows}
    actual_images = {str(path.relative_to(package)) for batch in BATCHES for path in (package / batch).glob('*.png')}
    if actual_images != expected_images:
        raise ValueError('Folder images differ from manifest')
    for batch in BATCHES:
        for path in (package / batch).glob('*.json'):
            if not path.with_suffix('.png').exists():
                raise ValueError(f'Orphan annotation: {path}')
    return result


def counts(package, rows):
    output = {}
    for batch in BATCHES:
        selected = [r for r in rows if r['annotation_batch'] == batch]
        labels = [package / r['expected_annotation_json'] for r in selected]
        output[batch] = {'n': len(selected),
                         'classes': dict(Counter(r['distance_label'] for r in selected)),
                         'videos': dict(Counter(r['video_id'] for r in selected)),
                         'json_n': sum(p.exists() for p in labels),
                         'missing_json_n': sum(not p.exists() for p in labels)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='annotation_projects/forceps_head_pilot_20260907')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--revision', default='mixed_20260917')
    args = parser.parse_args()
    package = ROOT / args.package
    history = package / 'split_history' / args.revision
    if history.exists():
        raise FileExistsError(f'Revision already exists; refusing another exchange: {history}')
    metadata_names = ('manifest.csv', 'annotation_status.csv', 'selection_audit.json', 'README.md')
    originals = {name: (package / name).read_bytes() for name in metadata_names}
    rows = read_csv(package / 'manifest.csv')
    statuses = read_csv(package / 'annotation_status.csv')
    if {r['sample_id'] for r in statuses} != {r['sample_id'] for r in rows} or len(statuses) != len(rows):
        raise ValueError('Status table does not match manifest')
    before = inventory(package, rows)
    before_counts = counts(package, rows)
    if [before_counts[b]['n'] for b in BATCHES] != [30, 114, 36]:
        raise ValueError('Expected batches of 30/114/36 images')
    annotated = [r for r in rows if before[r['sample_id']]['annotation_sha256'] is not None]
    train = sorted((r for r in annotated if r['annotation_batch'] != '03_validation'), key=lambda r: r['sample_id'])
    val = sorted((r for r in annotated if r['annotation_batch'] == '03_validation'), key=lambda r: r['sample_id'])
    rng = random.Random(args.seed)
    val_selected = rng.sample(val, 24)
    train_selected = rng.sample(train, 24)
    updated = copy.deepcopy(rows)
    by_id = {r['sample_id']: r for r in updated}
    moves = []
    exchanges = []
    for index, (v, t) in enumerate(zip(val_selected, train_selected), 1):
        for source, destination_batch in ((v, t['annotation_batch']), (t, '03_validation')):
            old = Path(source['annotation_image'])
            new = Path(destination_batch) / old.name
            for suffix in ('.png', '.json'):
                src, dst = old.with_suffix(suffix), new.with_suffix(suffix)
                if (package / dst).exists():
                    raise FileExistsError(package / dst)
                moves.append({'source': str(src), 'destination': str(dst), 'sha256': sha(package / src)})
            record = by_id[source['sample_id']]
            record.update(annotation_batch=destination_batch, annotation_image=str(new), expected_annotation_json=str(new.with_suffix('.json')))
            exchanges.append({'pair': index, 'sample_id': source['sample_id'], 'video_id': source['video_id'],
                              'distance_label': source['distance_label'], 'old_batch': source['annotation_batch'],
                              'new_batch': destination_batch, 'old_image': str(old), 'new_image': str(new),
                              'old_json': str(old.with_suffix('.json')), 'new_json': str(new.with_suffix('.json'))})
    for row in updated:
        row['annotation_split'] = 'val' if row['annotation_batch'] == '03_validation' else 'train'
    updated_status = copy.deepcopy(statuses)
    for row in updated_status:
        row['annotation_image'] = by_id[row['sample_id']]['annotation_image']
    plan = {'status': 'planned', 'seed': args.seed, 'revision': args.revision,
            'selection': 'Uniform sampling without replacement from image/JSON pairs; empty JSONs eligible; no class or per-folder quotas',
            'eligible_train_n': len(train), 'eligible_val_n': len(val),
            'exchange_pairs': 24, 'moves': moves, 'before_counts': before_counts,
            'before_inventory': before}
    history.mkdir(parents=True)
    for name, data in originals.items():
        (history / name).write_bytes(data)
    # Preserve all current manual labels independently of the move log.
    for row in rows:
        path = package / row['expected_annotation_json']
        if path.exists():
            backup = history / 'annotations_before' / row['expected_annotation_json']
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
    (history / 'exchange.csv').write_bytes(csv_bytes(exchanges))
    (history / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    completed = []
    try:
        for name, data in originals.items():
            if (package / name).read_bytes() != data:
                raise RuntimeError(f'Metadata changed during preparation: {name}')
        for move in moves:
            src, dst = package / move['source'], package / move['destination']
            if sha(src) != move['sha256'] or dst.exists():
                raise RuntimeError(f'File changed during preparation: {src}')
            src.rename(dst)
            completed.append(move)
        after = inventory(package, updated)
        if after != before:
            raise RuntimeError('Image or annotation content changed')
        after_counts = counts(package, updated)
        if [after_counts[b]['n'] for b in BATCHES] != [30, 114, 36]:
            raise RuntimeError('Batch size changed')
        audit = json.loads(originals['selection_audit.json'])
        audit.setdefault('initial_batch_counts', audit['batch_counts'])
        audit.update(batch_counts=after_counts, current_split_revision=args.revision,
                     current_split_protocol='User-requested random image-level exchange; videos overlap between train and validation',
                     current_split_field='manifest.csv: annotation_split; original_split is source provenance only',
                     exchange_record=f'split_history/{args.revision}/exchange.csv')
        atomic_write(package / 'manifest.csv', csv_bytes(updated))
        atomic_write(package / 'annotation_status.csv', csv_bytes(updated_status))
        atomic_write(package / 'selection_audit.json', json.dumps(audit, ensure_ascii=False, indent=2).encode('utf-8'))
        readme = originals['README.md'].decode('utf-8')
        readme = readme.replace('先确认头部边界与标注口径 | 10 / 10 / 10',
                                '训练子集（已混合交换） | ' + class_counts(after_counts[BATCHES[0]]))
        readme = readme.replace('口径确定后继续标注 | 38 / 38 / 38',
                                '训练子集（已混合交换） | ' + class_counts(after_counts[BATCHES[1]]))
        readme = readme.replace('独立验证视频，保留作验证 | 12 / 12 / 12',
                                '混合验证子集 | ' + class_counts(after_counts[BATCHES[2]]))
        readme = readme.replace('三个文件夹互不重复。前两个文件夹都属于原训练划分，可合并用于训练；验证图来自原验证视频 bend_data1。没有抽取测试图。',
                                '三个文件夹图像互不重复。前两个文件夹合并作为当前训练集；第三个作为当前验证集。按该实验混合协议划分后，训练和验证包含相同来源视频的不同帧，当前不属于按视频隔离的泛化评估。没有抽取测试图。')
        readme = readme.replace('后续训练应继续维持原来的视频划分，不能将验证图混入训练。',
                                '后续实验使用 manifest.csv 的 annotation_split 作为当前划分；original_split 仅记录图像原始来源，不可用它重新分配本包的训练和验证。原分类数据集的 CSV 没有修改。')
        notice = (f'## 当前划分：{args.revision}\n\n'
                  f'根据混合划分协议，从原验证文件夹交换 24 张图及 JSON 到训练文件夹，并从两个训练文件夹合计随机交换 24 张图及 JSON 到验证文件夹。随机种子 {args.seed}；仅在有 JSON 的图像中抽样，空标注参与抽样，无 JSON 的图像保留原位。\n\n'
                  f'当前仍为 144 张训练图和 36 张验证图，三个子文件夹仍为 30 / 114 / 36 张。交换明细和交换前清单、全部已有 JSON 备份位于 `split_history/{args.revision}/`。\n\n'
                  '旧的 `pilot_review_20260917` 页面是交换前的快照，不代表当前第一批成员。下文“抽样依据”描述最初选取 180 张图像的方式；随机交换后每个文件夹的类别数以当前表格为准。\n\n')
        readme = readme.replace('## 标注顺序\n', notice + '## 标注顺序\n', 1)
        atomic_write(package / 'README.md', readme.encode('utf-8'))
        report = ['# 标注数据集随机交换记录', '', f'状态：完成。随机种子：{args.seed}。', '',
                  '原验证文件夹 24 张与两个训练文件夹合计 24 张交换，共移动 48 张 PNG 和对应的 48 个 JSON。',
                  f'抽样池：有 JSON 的训练图 {len(train)} 张，有 JSON 的验证图 {len(val)} 张。空标注参与随机抽样；缺少 JSON 的样本留在原位置。', '',
                  '| 文件夹 | 图像 | 有 JSON | 无 JSON | TooFar / Good / TooClose |', '|---|---:|---:|---:|---|']
        for batch in BATCHES:
            info = after_counts[batch]
            report.append(f'| {batch} | {info["n"]} | {info["json_n"]} | {info["missing_json_n"]} | {class_counts(info)} |')
        report += ['', '## 划分语义与完整性', '',
                   '- `annotation_split` 为当前 train/val；`original_split` 保留原来源。',
                   '- 180 张图像和全部已有 JSON 的 SHA256 与交换前一致，图像／标注没有重复或丢失。',
                   '- 原始分类数据集 CSV、历史模型权重和历史实验结果未参与修改。',
                   '- 混合后训练与验证有同来源视频的不同帧，不能把结果解释为跨视频或未见器械类型泛化。',
                   '- 本次没有把缺失标注自动转换为空标注，也没有修改状态表中的人工判定。', '',
                   '## 追溯与恢复', '',
                   '- [逐样本交换记录](exchange.csv)', '- `plan.json`：逐文件原路径、新路径、指纹。',
                   '- 本目录下的 CSV、README、selection_audit.json 是交换前版本；annotations_before/ 保存交换前全部已有 JSON。',
                   '- 如需恢复，应先保存后续新增标注，按 plan.json 的 moves 逆序将 destination 移回 source，再恢复交换前清单；避免覆盖后续标注。', '']
        (history / 'report.md').write_text('\n'.join(report), encoding='utf-8')
        plan.update(status='complete', after_counts=after_counts, content_hashes_unchanged=True)
        atomic_write(history / 'plan.json', json.dumps(plan, ensure_ascii=False, indent=2).encode('utf-8'))
    except Exception as error:
        for move in reversed(completed):
            (package / move['destination']).rename(package / move['source'])
        for name, data in originals.items():
            (package / name).write_bytes(data)
        plan.update(status='rolled_back', error=str(error))
        (history / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    print(json.dumps({'revision': args.revision, 'after_counts': after_counts,
                      'train_to_val_by_folder': dict(Counter(r['annotation_batch'] for r in train_selected)),
                      'history': str(history), 'integrity': 'All image and annotation hashes unchanged'}, ensure_ascii=False, indent=2))


def class_counts(info):
    return ' / '.join(str(info['classes'].get(label, 0)) for label in ('TooFar', 'Good', 'TooClose'))


if __name__ == '__main__':
    main()
