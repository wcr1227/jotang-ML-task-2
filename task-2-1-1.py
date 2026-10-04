# 申明：代码由ai编写，疑问与注释由本人书写
# Pillow 读取图片文件
from PIL import Image
import numpy as np
import torch

# 1. 用 Pillow 打开图片
img = Image.open("task-2-1.png")
print("PIL 信息：")
print("  size =", img.size)     # (宽, 高)
print("  mode =", img.mode)     # RGB

# 2. 转成 NumPy 数组
arr = np.array(img)
print("\nNumPy 数组：")
print("  shape =", arr.shape)        # (高, 宽, 3)
print("  dtype =", arr.dtype)        # uint8,无符号8位整数，范围 0~255  0黑色，255白色
print("  min   =", arr.min())        # 0
print("  max   =", arr.max())        # 255

# 3. 打印一个像素的 RGB
y, x = 100, 200
print(f"\n像素 ({y}, {x}) 的 RGB =", arr[y, x])

# 4. 转成 PyTorch Tensor
# Numpy图像 HWC：shape=(H,W,C)  高 宽 通道
# PyTorch图像 CHW：shape=(C,H,W) 通道放到最前面，适配卷积运算
tensor = torch.from_numpy(arr).permute(2, 0, 1) # 维度重排 把HWC转换为CHW  
print("\nPyTorch Tensor:")
print("  shape =", tensor.shape)     # (3, 高, 宽)
print("  dtype =", tensor.dtype)     # torch.uint8
print("  min   =", tensor.min().item())
print("  max   =", tensor.max().item())