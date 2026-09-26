import numpy as np
from PIL import Image

def conv2d_gray(image, kernel, padding=0, stride=1):
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    if padding > 0:
        padded = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded = image
    output = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = padded[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    return output


# 读取图片
gray = np.array(Image.open("test.jpg").convert("L"))
print(f"输入尺寸：{gray.shape}")

kernel = np.ones((3, 3)) / 9  # 3×3 均值核

# valid：padding=0
result_valid = conv2d_gray(gray, kernel, padding=0, stride=1)
print(f"valid:  输入 {gray.shape} → 输出 {result_valid.shape}")

# same：padding=1
result_same = conv2d_gray(gray, kernel, padding=1, stride=1)
print(f"same:   输入 {gray.shape} → 输出 {result_same.shape}")

# 对比边缘像素
print(f"\nvalid 边缘像素：")
print(f"  左上角：{result_valid[0, 0]:.2f}")
print(f"  右下角：{result_valid[-1, -1]:.2f}")

print(f"\nsame 边缘像素：")
print(f"  左上角：{result_same[0, 0]:.2f}")
print(f"  右下角：{result_same[-1, -1]:.2f}")

# 计算边缘差异
print(f"\nvalid 边缘行均值：{result_valid[0].mean():.2f}")
print(f"same  边缘行均值：{result_same[0].mean():.2f}")