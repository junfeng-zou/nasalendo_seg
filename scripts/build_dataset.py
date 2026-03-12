import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ================= 配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POS_BASE = PROJECT_ROOT / "annotations" / "positive"
NEG_BASE = PROJECT_ROOT / "annotations" / "negative"

DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

TRAIN_RATIO = 0.85

# 正样本来源 (包含 preliminary 里的 output1/output2 和 formal 里的 output4/5/6)
POS_SOURCES = [
    POS_BASE / "output1",
    POS_BASE / "output2",
    POS_BASE / "formal" / "output4",
    POS_BASE / "formal" / "output5",
    POS_BASE / "formal" / "output6",
]

# 负样本来源 (比例抽取)
NEG_SOURCES = [
    NEG_BASE / "output1",
    NEG_BASE / "output2",
    NEG_BASE / "formal" / "output3",
    NEG_BASE / "formal" / "output4",
    NEG_BASE / "formal" / "output5",
    NEG_BASE / "formal" / "output6",
]

def clear_dataset():
    """清空原有的 dataset 目录"""
    if DATASET_ROOT.exists():
        print(f"清空原有目录: {DATASET_ROOT}")
        shutil.rmtree(DATASET_ROOT)
    
    # 重新创建必须的子目录
    for split in ["train", "val"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

def collect_files(directories):
    """收集目录下所有的图片及其对应的标签"""
    images = []
    labels = []
    
    for d in directories:
        if not d.exists():
            print(f"[警告] 目录不存在: {d}")
            continue
            
        for img_path in d.glob("*.png"):
            # 对应的 yolo txt 文件名
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                images.append(img_path)
                labels.append(txt_path)
                
    return images, labels

def distribute_dataset(images, labels, txt_prefix=""):
    """乱序并按照设定比例分配到 train 和 val"""
    # 绑定打乱
    combined = list(zip(images, labels))
    random.shuffle(combined)
    
    total = len(combined)
    train_count = int(total * TRAIN_RATIO)
    
    train_data = combined[:train_count]
    val_data = combined[train_count:]
    
    # 复制文件
    print(f"[{txt_prefix}] 总数: {total}, 分配到 train: {len(train_data)}, 分配到 val: {len(val_data)}")
    
    def copy_data(data, split):
        for i, (img_src, txt_src) in enumerate(data):
            # 重新命名，保证一一对应 (例如: Positives_train_00001.png)
            new_name = f"{txt_prefix}_{split}_{i:05d}"
            
            # 复制图片
            img_dst = IMAGES_DIR / split / f"{new_name}.png"
            shutil.copy2(img_src, img_dst)
            
            # 复制/创建标签
            txt_dst = LABELS_DIR / split / f"{new_name}.txt"
            
            # 严格确保所有的负样本标注为空文件
            if txt_prefix == "Negatives":
                with open(txt_dst, 'w', encoding='utf-8') as f:
                    pass # 置空
            else:
                shutil.copy2(txt_src, txt_dst)
            
    copy_data(train_data, "train")
    copy_data(val_data, "val")
    
    return len(train_data), len(val_data)

def main():
    # 1. 设置随机种子保证可重复性
    random.seed(42)
    
    # 2. 清空并重建目录
    clear_dataset()
    
    # 3. 收集所有目标正样本
    print("\n--- 收集正样本 ---")
    pos_images, pos_labels = collect_files(POS_SOURCES)
    num_pos = len(pos_images)
    print(f"获取了 {num_pos} 个正样本。")
    if num_pos == 0:
        print("[错误] 未找到任何正样本，停止执行！")
        return
        
    # 4. 根据正样本数量决定负样本需求数量并比例抽样
    target_neg_count = num_pos // 3
    print(f"\n--- 收集负样本 ---")
    print(f"计划抽取负样本总数 (正样本的 1/3): {target_neg_count}")
    
    # 使用所有所有可能的负样本来源
    all_neg_images = []
    all_neg_labels = []
    
    # 获取各个负样本目录的文件
    neg_dir_files = {} # dict: path -> [(img, txt)]
    total_avail_neg = 0
    
    for ndir in NEG_SOURCES:
        if not ndir.exists():
            continue
        imgs, txts = collect_files([ndir])
        pairs = list(zip(imgs, txts))
        if pairs:
            # 提前打乱这个组的顺序，保证比例抽样没有偏差
            random.shuffle(pairs)
            neg_dir_files[ndir] = pairs
            total_avail_neg += len(pairs)
            
    print(f"总计可用的负样本池: {total_avail_neg}")
    if target_neg_count > total_avail_neg:
        print(f"[警告] 目标负样本数 ({target_neg_count}) 大于可用总数 ({total_avail_neg})！将使用所有可用负样本。")
        target_neg_count = total_avail_neg
        
    # 按比例抽取负样本
    sampled_neg = []
    for ndir, pairs in neg_dir_files.items():
        # 每个目录需要抽取的数量 = (该目录可用数 / 总可用数) * 目标抽取数
        dir_sample_count = int((len(pairs) / total_avail_neg) * target_neg_count)
        sampled_neg.extend(pairs[:dir_sample_count])
        print(f"  从 {ndir.name} 中抽取 {dir_sample_count} / {len(pairs)} 个")
        
    # 由于由于整除取整，可能会有一点数量偏差，需要补齐或丢弃多余的
    diff = target_neg_count - len(sampled_neg)
    if diff > 0:
        # 在剩下的未抽取样本中随机补齐
        remainders = []
        for ndir, pairs in neg_dir_files.items():
            dir_sample_count = int((len(pairs) / total_avail_neg) * target_neg_count)
            remainders.extend(pairs[dir_sample_count:])
        random.shuffle(remainders)
        sampled_neg.extend(remainders[:diff])
        print(f"  通过零散补齐 {diff} 个样本使总数达到 {target_neg_count}")
        
    # 将打包的 list 解开
    if sampled_neg:
        neg_images, neg_labels = zip(*sampled_neg)
        neg_images, neg_labels = list(neg_images), list(neg_labels)
    else:
        neg_images, neg_labels = [], []
        
    print(f"实际抽取负样本数: {len(neg_images)}")
    
    # 5. 打乱分配到 train / val (正负样本独立打乱确保比例均衡)
    print("\n--- 分配正样本到 Train/Val ---")
    pos_train_num, pos_val_num = distribute_dataset(pos_images, pos_labels, "Positives")
    
    print("\n--- 分配负样本到 Train/Val ---")
    neg_train_num, neg_val_num = distribute_dataset(neg_images, neg_labels, "Negatives")
    
    # 6. 总结
    print(f"\n==============================================")
    print(f"数据集构建完成！总包含 {len(pos_images) + len(neg_images)} 张图片")
    print(f"- 训练集 (Train) 共 {pos_train_num + neg_train_num} 张 (正: {pos_train_num}, 负: {neg_train_num})")
    print(f"- 验证集 (Val) 共 {pos_val_num + neg_val_num} 张 (正: {pos_val_num}, 负: {neg_val_num})")
    print(f"比例 Train/Val 约: {(pos_train_num+neg_train_num)/(len(pos_images)+len(neg_images)):.2f} / {(pos_val_num+neg_val_num)/(len(pos_images)+len(neg_images)):.2f}")
    print(f"==============================================")


if __name__ == "__main__":
    main()
