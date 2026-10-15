# 申明：代码由ai辅助书写，学习代码过程中的疑问与解答由本人书写
# ============================================================
# 猫狗分类器（Kaggle 数据集 + 小数据集版）- 完整实践代码 (PyTorch)
# ============================================================
# 【这次一共改了什么】(下面代码里都用 "【这里改了什么】" 标出来了)
#   1. 注释掉 prepare_data() 的调用    —— Kaggle 数据集已经是 cats/、dogs/ 两个文件夹，不用再重组
#   2. 删掉自定义的 CatDogDataset 类   —— 改用 torchvision 自带的 ImageFolder
#   3. 数据加载改成 ImageFolder + DataLoader，路径指向 make_small_data.py 生成的 small-data/
#   4. 删掉测试集 test_dataset / test_loader —— Kaggle 的 test_set 没有标签，没法算准确率
#      最后的评估改成用验证集 val_loader
#   5. 类别名改成复数 ['cats', 'dogs']  —— 和文件夹名字保持一致
#   6. 超参数按 CPU / GPU 各给一套建议值 —— CPU: 128 + batch 16；GPU: 224 + batch 64
# ============================================================

import os
import glob
import random
import shutil # 文件操作（现在只在下面被注释掉的 prepare_data 里还会用到）
import numpy as np
import matplotlib.pyplot as plt # 画图
import torch
import torch.nn as nn
import torch.nn.functional as F # 损失函数
import torch.optim as optim # 优化器
from torch.utils.data import DataLoader # 【这里改了什么】不再需要自定义 Dataset / random_split
from torchvision import transforms # 图片预处理
from torchvision.datasets import ImageFolder # 【这里改了什么】官方自带的"按文件夹分类"数据集
from PIL import Image

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设备配置：有显卡就用显卡（cuda），没有就用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    # 打印显卡名字，确认真的用上了 GPU
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch 版本: {torch.__version__}  CUDA 版本: {torch.version.cuda}")
else:
    print("没有检测到可用 GPU（或者装的是 CPU 版 PyTorch），本次全部用 CPU 跑")

# ============================================================
# 0. 【这里改了什么】超参数：CPU 和 GPU 各一套建议值
# ============================================================
# 参数        CPU                 GPU                  为什么这么定？
# ----------  ------------------  -------------------  ------------------------------
# IMG_SIZE    128                 224                  CPU 用小图，算得快；GPU 用 224 精度更高
# BATCH_SIZE  16                  64                   显卡显存大，一次可以喂更多图片
# NUM_WORKERS 0(Windows)/4(Linux) 4(Windows 报错改 0)   Windows 上多进程读图容易报错，默认给 0
# EPOCHS      5~10                15                   CPU 每轮都慢，先跑少一点看趋势
USE_GPU = torch.cuda.is_available() # 想强制用 CPU 跑（比如对比速度），把这里改成 False

if USE_GPU:
    IMG_SIZE, BATCH_SIZE, EPOCHS = 224, 64, 15
else:
    IMG_SIZE, BATCH_SIZE, EPOCHS = 128, 16, 8

# num_workers = 用几个"小工"提前把图片读进内存。
#   Windows：子进程用 spawn 方式启动，num_workers>0 很容易报错，所以默认 0（最保险）
#   Linux / Mac：给 4 能明显加快读图速度
#   如果 Windows 上你也想用 4，需要把下面的主流程放进 if __name__ == '__main__': 里（见课后说明）
NUM_WORKERS = 0 if os.name == 'nt' else 4

print(f"\n本次配置: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
      f"EPOCHS={EPOCHS}, NUM_WORKERS={NUM_WORKERS}")

# ============================================================
# 1. 数据集准备与划分
# ============================================================

# --- 1.1 数据清洗与目录重组 ---
# 整理猫狗图片数据集：把杂乱放在一起的猫、狗图片，重新整理文件夹，并且按7:1.5:1.5分成训练集、验证集、测试集
def prepare_data(raw_dir='train', output_dir='data'):
    #raw_dir：原始文件夹
    #output_dir：整理好之后输出到 data 文件夹
    """
    将原始图片目录重组为 ImageFolder 所需的分类结构
    并划分 train/val/test
    输出结构:
        data/train/cat/, data/train/dog/
        data/val/cat/,   data/val/dog/
        data/test/cat/,  data/test/dog/
    """
    # 判断是否已经整理好
    if os.path.exists(os.path.join(output_dir, 'train')):
        print(f"数据目录 {output_dir} 已存在，跳过重组")
        return

    # 创建目录子文件夹
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # 收集所有图片并按类别分组
    # glob:批量查找所有猫图片、狗图片，统计数量
    cat_files = glob.glob(os.path.join(raw_dir, 'cat.*.jpg'))
    dog_files = glob.glob(os.path.join(raw_dir, 'dog.*.jpg'))
    print(f"原始数据: Cat={len(cat_files)}, Dog={len(dog_files)}")

    # 打乱并划分 (7:1.5:1.5)
    def split_files(files):
        random.shuffle(files) # 打乱顺序
        n = len(files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

    cat_train, cat_val, cat_test = split_files(cat_files) # 分三份
    dog_train, dog_val, dog_test = split_files(dog_files)

    # 复制文件
    splits = {
        'train': {'cat': cat_train, 'dog': dog_train},
        'val':   {'cat': cat_val,   'dog': dog_val},
        'test':  {'cat': cat_test,  'dog': dog_test},
    }
    # 三层循环
    # 第一层：先选定训练集还是验证集测试集
    for split_name, cls_dict in splits.items():
        # 第二层：猫还是狗
        for cls_name, files in cls_dict.items():
            # 第三层：遍历每张图片
            for f in files:
                dst = os.path.join(output_dir, split_name, cls_name, os.path.basename(f))
                shutil.copy(f, dst)
        print(f"{split_name} 集完成: Cat={len(splits[split_name]['cat'])}, Dog={len(splits[split_name]['dog'])}")

# 【这里改了什么】这一行被注释掉了！
# 原来的数据集是"平铺"结构（train/cat.1.jpg、train/dog.1.jpg），所以要先用 prepare_data()
# 把它们整理成 train/cat/、train/dog/ 才能训练。
# 现在用的是 Kaggle 数据集，它本身就已经是 training_set/training_set/cats|dogs 这种文件夹结构，
# 不需要再整理；而且这次我们只用一小部分数据，挑选和复制交给 make_small_data.py 完成。
# prepare_data(raw_dir='train', output_dir='data')

# --- 1.2 【这里改了什么】删掉了自定义的 CatDogDataset 类，改用 ImageFolder ---
# 原来的写法是自己写一个 Dataset 类，一行行读文件夹、记路径、记标签。
# torchvision 已经自带了一个干这件事的类：ImageFolder。
# 它要求目录长这样：
#       small-data/train/cats/*.jpg
#       small-data/train/dogs/*.jpg
# 它会自动扫描所有图片，并且"按文件夹名"生成标签：
#       classes = ['cats', 'dogs']   ->   cats 标签是 0，dogs 标签是 1
# 注意：ImageFolder 是按文件夹名字母顺序编号的，cats 在 dogs 前面，所以 cats=0、dogs=1。

# ============================================================
# 2. 数据预处理与 DataLoader
# ============================================================
# IMG_SIZE 已经在上面按 CPU/GPU 定好了：CPU 128，GPU 224
# ImageNet 统计量，用于预训练模型的标准归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 训练集：数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # 图片缩放
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10), # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),# 随机微调亮度、对比度、饱和度，模拟不同光照下拍的照片
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),# 减去均值，除以方差，完成标准化，适配预训练模型
])

# 验证集：不做随机增强（要稳定，才能公平地比较每一轮的好坏）
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 【这里改了什么】数据加载改用 ImageFolder（先运行 make_small_data.py 生成 small-data/）
# 小提示：这是相对路径，所以要"在 small-data 的上一级目录"运行本脚本；
#         或者改成绝对路径，例如：
#         ImageFolder(r'D:\Code\Jotang-ml-task2\small-data\train', transform=train_transform)
train_dataset = ImageFolder('small-data/train', transform=train_transform)
val_dataset   = ImageFolder('small-data/val',   transform=val_transform)

# 【这里改了什么】原来还有一行 test_dataset（测试集），现在删掉了

CLASS_NAMES = train_dataset.classes   # ImageFolder 自动读出来的类别名：['cats', 'dogs']
print(f"类别名: {CLASS_NAMES}   对应编号: {train_dataset.class_to_idx}")
print(f"训练集 {len(train_dataset)} 张，验证集 {len(val_dataset)} 张")

# DataLoader 数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
# shuffle=True （训练集）：每一轮 epoch 打乱图片顺序，防止模型死记图片顺序
# shuffle=False（验证集）：不需要打乱，保证评估稳定
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 【这里改了什么】原来还有 test_loader，也删掉了。
# 原因：Kaggle 的 test_set 只有图片、没有标签，算不出准确率，
#       所以最终评估改用验证集 val_loader（既能看效果，又不会"偷看"标签）。

# 检查一个 batch 的形状
imgs, labels = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")

# ============================================================
# 图片显示小工具
# ============================================================
# 每张图除了弹窗显示，还顺手存一份 PNG 到 outputs/ 文件夹，方便写作业截图（不需要可以删掉这两行）
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def show_fig(fig, filename=None):
    """显示图片；给了文件名就顺便保存到 outputs/ 文件夹"""
    if filename:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=110, bbox_inches='tight')
    plt.show()

# --- 可视化预处理前后的图片 ---
def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

def visualize_preprocessing(raw_path):
    img_raw = Image.open(raw_path).convert('RGB')
    img_tensor = val_transform(img_raw)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_raw)
    axes[0].set_title(f'Before: {img_raw.size}')
    axes[0].axis('off')

    axes[1].imshow(denormalize(img_tensor).permute(1, 2, 0))
    axes[1].set_title(f'After: {tuple(img_tensor.shape)}')
    axes[1].axis('off')
    show_fig(fig, '1_预处理前后.png')

# 从训练集里随便找一张展示
# ImageFolder 的 .samples 是一个列表，每个元素是 (图片路径, 标签)，例如 ('small-data/train/cats/cat.1.jpg', 0)
sample_path = train_dataset.samples[0][0]
visualize_preprocessing(sample_path)

# ============================================================
# 3. 搭建 CNN 模型
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 输入: (3, IMG_SIZE, IMG_SIZE)，比如 CPU 模式是 (3, 128, 128)
        # 不同的卷积核提取不同层次的图片特征
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> (32, IMG_SIZE, IMG_SIZE)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)                              # -> (32, IMG_SIZE/2, IMG_SIZE/2)
        # 卷积 + 归一化 + 池化

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, ...)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)                              # -> (64, ...)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, ...)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)                              # -> (128, ...)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # -> (256, ...)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)                               # -> (256, ...)

        self.gap = nn.AdaptiveAvgPool2d(1) # 全局平均池化           # -> (256, 1, 1)
        # 好处：不管输入图片是 128 还是 224，这里都能压成 1×1，所以同一个模型两种尺寸都能跑
        self.dropout = nn.Dropout(0.5)
        # 随机关掉50%神经元，防止过拟合（为什么？感觉ai解释的很抽象）
        self.fc = nn.Linear(256, num_classes) # 全连接层 256个特征 -> 2个类别（cats, dogs）

    # 前向传播
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

model = SimpleCNN(num_classes=2).to(device)
print(model)

# --- 3.1 记录各层 Tensor 形状 ---
def print_layer_shapes(model, input_tensor):
    print("\n--- Layer Shapes ---")
    print(f"Input:        {tuple(input_tensor.shape)}")
    x = input_tensor
    x = F.relu(model.bn1(model.conv1(x))); print(f"conv1+bn+relu: {tuple(x.shape)}")
    x = model.pool1(x);                    print(f"pool1:         {tuple(x.shape)}")
    x = F.relu(model.bn2(model.conv2(x))); print(f"conv2+bn+relu: {tuple(x.shape)}")
    x = model.pool2(x);                    print(f"pool2:         {tuple(x.shape)}")
    x = F.relu(model.bn3(model.conv3(x))); print(f"conv3+bn+relu: {tuple(x.shape)}")
    x = model.pool3(x);                    print(f"pool3:         {tuple(x.shape)}")
    x = F.relu(model.bn4(model.conv4(x))); print(f"conv4+bn+relu: {tuple(x.shape)}")
    x = model.pool4(x);                    print(f"pool4:         {tuple(x.shape)}")
    x = model.gap(x);                      print(f"GAP:           {tuple(x.shape)}")
    x = x.flatten(1);                      print(f"Flatten:       {tuple(x.shape)}")

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
print_layer_shapes(model, dummy_input)

# ============================================================
# 4. 训练与验证
# ============================================================

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5) #学习率调度器

# EPOCHS 已经在上面按 CPU/GPU 定好了
best_val_acc = 0.0 # 记录目前最好的验证集准确率，用来保存最优模型
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device) # 把图片、标签送到GPU（device）加速计算
        optimizer.zero_grad() # 清空上一轮保存的梯度
        outputs = model(imgs) # 正向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion) # 【这里改了什么】验证集
    scheduler.step(val_loss)

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);   val_accs.append(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # 保存最佳模型（非常重要！！！）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(), # 网络权重，卷积核参数
            'img_size': IMG_SIZE,
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD,
            'class_names': CLASS_NAMES, # 【这里改了什么】改成 ['cats', 'dogs']，和文件夹名字一致
            'best_val_acc': best_val_acc,
        }, 'best_model.pth')
        print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")

# --- 绘制 Loss 和 Accuracy 曲线 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(train_accs, label='Train Acc')
axes[1].plot(val_accs, label='Val Acc')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curve'); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); show_fig(fig, '2_训练曲线.png')

# ============================================================
# 5. 【这里改了什么】最终评估：原来用测试集，现在用验证集
# ============================================================

checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# 【这里改了什么】原来这里是 test_loader，现在改成 val_loader
# （Kaggle 的 test_set 没有标签，只能用带标签的验证集来做最终评估）
val_loss, val_acc = validate(model, val_loader, criterion)
print(f"\n[最终评估] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ============================================================
# 6. 特征图可视化
# ============================================================
# 直观看到卷积到底提取了图片什么信息
def visualize_feature_maps(model, img_tensor, layer_name='conv1', num_maps=8, save_name=None):
    features = {}
    # 钩子函数
    # hook 可以在网络正向传播经过某一层的时候，把这一层的输出截下来
    # 这段整体感觉有点没看懂（呜呜）
    def hook(module, input, output):
        features['out'] = output.detach()

    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        model(img_tensor.unsqueeze(0).to(device))
    handle.remove()

    fmap = features['out'][0].cpu()  # (C, H, W)
    num_maps = min(num_maps, fmap.shape[0])

    fig, axes = plt.subplots(2, num_maps // 2, figsize=(16, 5))
    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(fmap[i], cmap='viridis')
            ax.set_title(f'ch {i}')
        ax.axis('off')
    plt.suptitle(f'Feature maps of {layer_name}')
    show_fig(fig, save_name)

# 【这里改了什么】选一张验证集图片来画（原来用的是测试集图片）
val_img, val_label = val_dataset[0]
print(f"拿来做特征图可视化的图片: {val_dataset.samples[0][0]}，标签 = {CLASS_NAMES[val_label]}")
visualize_feature_maps(model, val_img, 'conv1', save_name='3_conv1特征图.png') # 特征图像原图
visualize_feature_maps(model, val_img, 'conv4', save_name='4_conv4特征图.png') # 特征图很抽象

# ============================================================
# 7. 数据增强
# ============================================================
# 数据增强：在训练时，对图片做随机变换，生成更多"新图片"，扩充训练集，减少过拟合
aug_transforms = {
    'none':      transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]),
    'flip':      transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    'crop':      transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(IMG_SIZE), transforms.ToTensor()]),
    'color':     transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.ToTensor()]),
    'combo':     transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(IMG_SIZE),
                                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.ToTensor()]),
}


def show_augmentations(img_path, transforms_dict):
    img = Image.open(img_path).convert('RGB')
    fig, axes = plt.subplots(1, len(transforms_dict), figsize=(4 * len(transforms_dict), 4))
    for ax, (name, tf) in zip(axes, transforms_dict.items()):
        out = tf(img)
        # 对于未归一化的 ToTensor 输出，直接显示
        ax.imshow(out.permute(1, 2, 0).numpy())
        ax.set_title(name)
        ax.axis('off')
    show_fig(fig, '5_数据增强.png')

# 【这里改了什么】改成从 train_dataset.samples 里取路径（ImageFolder 提供的）
sample_path = train_dataset.samples[0][0]
show_augmentations(sample_path, aug_transforms)

# ============================================================
# 8. 错误样本分析
# ============================================================
# 错误样本是干嘛用的？
# 通过错误样本，可以分析模型的短板，比如模型学习到了背景而不是动物本身，这时候就可以用数据增强来改善
def show_misclassified(model, loader, n=8):
    model.eval()
    wrong = [] # 存放预测出错样本
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    wrong.append((imgs[i].cpu(), labels[i].item(), preds[i].item()))
            if len(wrong) >= n:
                break

    n = min(n, len(wrong))
    if n == 0:
        print("No misclassified samples found!")
        return

    # 排成 2 行展示（顺手修个小坑：错误样本不够 2 张时，原来的写法会报错）
    rows = 2 if n >= 2 else 1
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, (img, true, pred) in zip(np.atleast_1d(axes).flat, wrong[:n]):
        ax.imshow(denormalize(img).permute(1, 2, 0)) # 反归一化，还原色彩
        ax.set_title(f'True: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]}') # 【这里改了什么】用 CLASS_NAMES
        ax.axis('off')
    plt.tight_layout(); show_fig(fig, '6_错误样本.png')

# 【这里改了什么】原来传的是 test_loader，现在传 val_loader
show_misclassified(model, val_loader, n=8)

# ============================================================
# 9. 独立推理程序
# ============================================================

def predict_image(img_path, checkpoint_path='best_model.pth'): # 用我们保存好的模型去预测
    """
    输入一张本地图片，输出预测类别、概率，并显示处理后的图片。
    """
    # 加载模型权重
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 图像预处理
    tf = transforms.Compose([
        transforms.Resize((ckpt['img_size'], ckpt['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt['mean'], ckpt['std']),
    ])

    # 读图转张量
    img_orig = Image.open(img_path).convert('RGB')
    img_tensor = tf(img_orig).unsqueeze(0).to(device)

    # 向前推理
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu() # softmax 算概率

    # 获取预测类别和置信度
    # 置信度：模型对自己这个猜测有多大把握，用百分比表示
    pred_idx = probs.argmax().item()
    pred_label = ckpt['class_names'][pred_idx] # 【这里改了什么】这里会是 'cats' 或 'dogs'
    confidence = probs[pred_idx].item()

    # 画图展示
    show_img = denormalize(img_tensor[0].cpu()).permute(1, 2, 0)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(show_img)
    plt.title(f'Prediction: {pred_label} ({confidence:.2%})')
    plt.axis('off')
    show_fig(fig)

    print(f"预测: {pred_label} | 概率: {confidence:.4f}")
    return pred_label, confidence

# 可替换图片路径（下面这张来自 Kaggle 原始训练集，想试就把注释去掉）
# predict_image('archive/training_set/training_set/cats/cat.1.jpg')
# predict_image('archive/training_set/training_set/dogs/dog.1.jpg')
