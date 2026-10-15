# -*- coding: utf-8 -*-
"""
make_small_data.py
======================================================================
作用：从 Kaggle 的猫狗数据集（training_set/training_set/cats、dogs）里，
      猫、狗各随机挑一部分图片，按 8:2 分成 train / val，
      复制到 small-data/ 文件夹，供 task-2-2.py（用 ImageFolder）直接训练。

为什么要单独写这个脚本？
    Kaggle 训练集有 8000+ 张（cats 4000 + dogs 4000），用 CPU 训练太慢；
    先挑出每类 500 张，几分钟就能跑完一轮。
    而且 ImageFolder 要求"一个类别一个文件夹"，这个脚本顺便把目录结构做好。

最终目录结构：
    small-data/
    ├── train/
    │   ├── cats/   （默认 400 张）
    │   └── dogs/   （默认 400 张）
    └── val/
        ├── cats/   （默认 100 张）
        └── dogs/   （默认 100 张）

用法（在 task-2-2.py 所在的目录打开终端执行）：
    python make_small_data.py
    # 想改数量 / 比例 / 路径就加参数：
    python make_small_data.py --per-class 300 --val-ratio 0.2 --src archive --dst small-data
======================================================================
"""

import argparse
import os
import random
import shutil
import sys

# 只认这些后缀的图片文件（数据集里可能混着 _DS_Store 之类的杂文件，会被自动忽略）
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# 固定随机种子：每次运行挑出来的图片完全一样，方便复现结果
SEED = 42

# 类别名（一定要和文件夹名字一致；ImageFolder 会把它们按字母顺序编号：cats=0, dogs=1）
CLASS_NAMES = ['cats', 'dogs']


def find_class_dir(src_root, class_name):
    """
    在 Kaggle 数据集里找 cats / dogs 文件夹。

    不同版本的 Kaggle 数据集解压后层级不太一样，
    这里把常见的几种写法都试一遍，找到就用，省得你手动改路径。
    """
    # cats -> cat、Cat；dogs -> dog、Dog（老版本数据集里的命名）
    single = class_name[:-1]
    candidates = [
        os.path.join(src_root, class_name),                                    # archive/cats
        os.path.join(src_root, 'training_set', class_name),                    # archive/training_set/cats
        os.path.join(src_root, 'training_set', 'training_set', class_name),    # archive/training_set/training_set/cats
        os.path.join(src_root, class_name.capitalize()),                       # archive/Cats、archive/Dogs
        os.path.join(src_root, 'PetImages', single.capitalize()),              # 老版本：PetImages/Cat、PetImages/Dog
        os.path.join(src_root, single.capitalize()),                           # archive/Cat、archive/Dog
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def list_images(folder):
    """列出文件夹里所有图片文件名，并按名字排序（排序是为了让结果可复现）"""
    names = [n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)]
    names.sort()   # 先排序，再用固定种子的随机数打乱 -> 每次结果都一样
    return names


def copy_files(names, src_dir, dst_dir):
    """把 names 里的图片从 src_dir 复制到 dst_dir，返回这次实际复制的张数"""
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for i, name in enumerate(names, start=1):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        # 已经复制过（而且大小一样）就跳过，重复运行不会白等一遍
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        shutil.copy2(src, dst)   # copy2 会连文件的修改时间一起复制
        copied += 1
    return copied


def make_small_data(src_root, dst_root, per_class, val_ratio, seed):
    """主流程：挑图片 -> 划分 train/val -> 复制 -> 打印数量"""
    print('=' * 64)
    print('原始数据集目录 :', os.path.abspath(src_root))
    print('小数据集输出到 :', os.path.abspath(dst_root))
    print(f'每类抽取       : {per_class} 张   （验证集占 {val_ratio:.0%}）')
    print('=' * 64)

    # 用固定的随机种子生成一个随机数对象：
    # 同一个种子 -> 每次打乱的结果都一样 -> 小数据集每次挑到的图片相同
    rng = random.Random(seed)

    summary = []   # 记录每个类别的统计信息，最后统一打印

    for class_name in CLASS_NAMES:
        src_dir = find_class_dir(src_root, class_name)
        if src_dir is None:
            sys.exit(f'[错误] 在 "{src_root}" 下面找不到 {class_name} 文件夹。\n'
                     f'       请用 --src 指定数据集所在目录，例如：\n'
                     f'       python make_small_data.py --src "D:\\Code\\Jotang-ml-task2\\archive"')

        all_images = list_images(src_dir)
        if len(all_images) < per_class:
            print(f'[提醒] {class_name} 一共只有 {len(all_images)} 张，'
                  f'少于要抽取的 {per_class} 张，那就全部用上')

        n_pick = min(per_class, len(all_images))

        rng.shuffle(all_images)          # 随机打乱
        picked = all_images[:n_pick]     # 取前 n_pick 张

        # 按比例切分：前面当训练集，后面当验证集
        n_val = int(round(n_pick * val_ratio))
        n_train = n_pick - n_val
        train_files = picked[:n_train]
        val_files = picked[n_train:]

        train_dir = os.path.join(dst_root, 'train', class_name)
        val_dir = os.path.join(dst_root, 'val', class_name)
        copy_files(train_files, src_dir, train_dir)
        copy_files(val_files, src_dir, val_dir)

        print(f'\n[{class_name}]')
        print(f'  来源目录 : {src_dir}')
        print(f'  原图总数 : {len(all_images)} 张')
        print(f'  抽出来的 : {n_pick} 张  ->  train {n_train} 张 + val {n_val} 张')
        print(f'  写到     : {split_hint(train_dir)}  和  {split_hint(val_dir)}')

        summary.append((class_name, len(all_images), n_train, n_val))

    # ---------------- 打印每类数量 ----------------
    print('\n' + '=' * 64)
    print('小数据集统计')
    print('-' * 64)
    total_train = total_val = 0
    for class_name, n_all, n_train, n_val in summary:
        print(f'  {class_name:<6}: 原图 {n_all} 张 -> train {n_train} 张, '
              f'val {n_val} 张  (小计 {n_train + n_val} 张)')
        total_train += n_train
        total_val += n_val
    print(f'  {"合计":<6}: train {total_train} 张, val {total_val} 张  '
          f'(总计 {total_train + total_val} 张)')
    print('=' * 64)

    # ---------------- 复查一遍：数一数输出目录里到底有多少张 ----------------
    print('\n复查输出目录（直接数文件个数）：')
    for split in ['train', 'val']:
        for class_name in CLASS_NAMES:
            folder = os.path.join(dst_root, split, class_name)
            n = len(list_images(folder)) if os.path.isdir(folder) else 0
            print(f'  {split}/{class_name}: {n} 张')

    print(f'\n完成！小数据集在: {os.path.abspath(dst_root)}')
    print('下一步：直接运行  python task-2-2.py  （它用 ImageFolder 读 small-data/train 和 small-data/val）')


def split_hint(path):
    """只是为了打印好看：把完整路径显示成 train/cats 这种简短写法"""
    parts = path.replace('\\', '/').split('/')
    return '/'.join(parts[-2:])


def main():
    parser = argparse.ArgumentParser(description='从 Kaggle 猫狗数据集里挑出一小份，做成 ImageFolder 结构')
    # --src：Kaggle 数据集解压后的根目录（默认 archive，也就是你机器上的 D:\\Code\\Jotang-ml-task2\\archive）
    parser.add_argument('--src', default='archive', help='原始数据集根目录，默认 archive')
    # --dst：小数据集输出目录
    parser.add_argument('--dst', default='small-data', help='输出目录，默认 small-data')
    # --per-class：每类抽多少张（抽完再按比例切 train/val）
    parser.add_argument('--per-class', type=int, default=500, help='每类抽取多少张，默认 500')
    # --val-ratio：验证集比例
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例，默认 0.2（也就是 8:2）')
    # --seed：随机种子
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子，默认 42')
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f'[错误] 找不到数据集目录 "{args.src}"。\n'
                 f'       请在 task-2-2.py 所在的目录运行，或用 --src 指定绝对路径。')

    make_small_data(
        src_root=args.src,
        dst_root=args.dst,
        per_class=args.per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
