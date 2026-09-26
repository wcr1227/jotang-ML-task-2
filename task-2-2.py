# ============================================================
# 猫狗分类器 - 完整实践代码 (PyTorch)
# ============================================================

import os
import glob
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
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

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. 数据集准备与划分
# ============================================================

# --- 1.1 数据清洗与目录重组 ---
def prepare_data(raw_dir='train', output_dir='data'):
    """
    将原始 flat 目录重组为 ImageFolder 所需的分类结构，
    并划分 train/val/test。
    原始目录: train/cat.0.jpg, train/dog.0.jpg, ...
    输出结构:
        data/train/cat/, data/train/dog/
        data/val/cat/,   data/val/dog/
        data/test/cat/,  data/test/dog/
    """
    if os.path.exists(os.path.join(output_dir, 'train')):
        print(f"数据目录 {output_dir} 已存在，跳过重组。")
        return

    # 创建目录
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # 收集所有图片并按类别分组
    cat_files = glob.glob(os.path.join(raw_dir, 'cat.*.jpg'))
    dog_files = glob.glob(os.path.join(raw_dir, 'dog.*.jpg'))
    print(f"原始数据: Cat={len(cat_files)}, Dog={len(dog_files)}")

    # 打乱并划分 (7:1.5:1.5)
    def split_files(files):
        random.shuffle(files)
        n = len(files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

    cat_train, cat_val, cat_test = split_files(cat_files)
    dog_train, dog_val, dog_test = split_files(dog_files)

    # 复制文件
    splits = {
        'train': {'cat': cat_train, 'dog': dog_train},
        'val':   {'cat': cat_val,   'dog': dog_val},
        'test':  {'cat': cat_test,  'dog': dog_test},
    }
    for split_name, cls_dict in splits.items():
        for cls_name, files in cls_dict.items():
            for f in files:
                dst = os.path.join(output_dir, split_name, cls_name, os.path.basename(f))
                shutil.copy(f, dst)
        print(f"{split_name} 集完成: Cat={len(splits[split_name]['cat'])}, Dog={len(splits[split_name]['dog'])}")

prepare_data(raw_dir='train', output_dir='data')

# --- 1.2 自定义 Dataset (更灵活，方便处理损坏图片) ---
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir 结构: root_dir/cat/*.jpg, root_dir/dog/*.jpg
        """
        self.samples = []
        self.transform = transform
        self.class_to_idx = {'cat': 0, 'dog': 1}

        for cls_name, label in self.class_to_idx.items():
            folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, fname), label))

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回一个全黑的占位图
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img, label

# ============================================================
# 2. 数据预处理与 DataLoader
# ============================================================

IMG_SIZE = 224
# ImageNet 统计量，用于预训练模型的标准归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 训练集：带数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 验证/测试集：仅 Resize + Normalize，不做随机增强
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 构建 Dataset
train_dataset = CatDogDataset('data/train', transform=train_transform)
val_dataset   = CatDogDataset('data/val',   transform=val_transform)
test_dataset  = CatDogDataset('data/test',  transform=val_transform)

# DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 检查一个 batch 的形状
imgs, labels = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")

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
    plt.show()

# 从训练集中随便找一张展示
sample_path = train_dataset.samples[0][0]
visualize_preprocessing(sample_path)

# ============================================================
# 3. 搭建 CNN 模型
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 输入: (3, 224, 224)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # -> (32, 224, 224)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)                              # -> (32, 112, 112)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 112, 112)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)                              # -> (64, 56, 56)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, 56, 56)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)                              # -> (128, 28, 28)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # -> (256, 28, 28)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)                               # -> (256, 14, 14)

        self.gap = nn.AdaptiveAvgPool2d(1)                         # -> (256, 1, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

EPOCHS = 15
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

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
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step(val_loss)

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);   val_accs.append(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'img_size': IMG_SIZE,
            'mean': IMAGENET_MEAN,
            'std': IMAGENET_STD,
            'class_names': ['cat', 'dog'],
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
plt.tight_layout(); plt.show()

# ============================================================
# 5. 测试集评估
# ============================================================

checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = validate(model, test_loader, criterion)
print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# ============================================================
# 6. 特征图可视化
# ============================================================

def visualize_feature_maps(model, img_tensor, layer_name='conv1', num_maps=8):
    features = {}
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
    plt.show()

# 选一张测试集图片
test_img, test_label = test_dataset[0]
visualize_feature_maps(model, test_img, 'conv1')
visualize_feature_maps(model, test_img, 'conv4')

# ============================================================
# 7. 数据增强对照实验（展示）
# ============================================================

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
    plt.show()

sample_path = train_dataset.samples[0][0]
show_augmentations(sample_path, aug_transforms)

# ============================================================
# 8. 错误样本分析
# ============================================================

def show_misclassified(model, loader, n=8):
    model.eval()
    wrong = []
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

    fig, axes = plt.subplots(2, n // 2, figsize=(4 * (n // 2), 8))
    for ax, (img, true, pred) in zip(axes.flat, wrong[:n]):
        ax.imshow(denormalize(img).permute(1, 2, 0))
        ax.set_title(f'True: {"cat" if true==0 else "dog"}\nPred: {"cat" if pred==0 else "dog"}')
        ax.axis('off')
    plt.tight_layout(); plt.show()

show_misclassified(model, test_loader, n=8)

# ============================================================
# 9. 独立推理程序
# ============================================================

def predict_image(img_path, checkpoint_path='best_model.pth'):
    """
    输入一张本地图片，输出预测类别、概率，并显示处理后的图片。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((ckpt['img_size'], ckpt['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt['mean'], ckpt['std']),
    ])

    img_orig = Image.open(img_path).convert('RGB')
    img_tensor = tf(img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    pred_idx = probs.argmax().item()
    pred_label = ckpt['class_names'][pred_idx]
    confidence = probs[pred_idx].item()

    # 显示
    show_img = denormalize(img_tensor[0].cpu()).permute(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(show_img)
    plt.title(f'Prediction: {pred_label} ({confidence:.2%})')
    plt.axis('off')
    plt.show()

    print(f"预测: {pred_label} | 概率: {confidence:.4f}")
    return pred_label, confidence

# 示例用法（替换为你自己的图片路径）
# predict_image('data/test/cat/cat.1234.jpg')