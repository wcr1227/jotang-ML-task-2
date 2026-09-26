import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==================== 核心：灰度图二维卷积 ====================

def conv2d_gray(image, kernel, padding=0, stride=1):
    """
    对灰度图进行二维卷积（核心手写实现）

    参数：
        image:  2D numpy array，形状 (H, W)
        kernel: 2D numpy array，形状 (kh, kw)
        padding: int，边缘补 0 的层数
        stride:  int，滑动步长

    返回：
        output: 2D numpy array，卷积结果
    """
    # 确保是浮点数，避免整数溢出
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    h, w = image.shape
    kh, kw = kernel.shape

    # 计算输出尺寸
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1

    # 如果尺寸不合法，报错
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"输出尺寸不合法：out_h={out_h}, out_w={out_w}。"
            f"请检查 padding/stride/kernel 大小。"
        )

    # 边缘填充
    if padding > 0:
        padded = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded = image

    # 初始化输出
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # 核心卷积：双重循环
    for i in range(out_h):
        for j in range(out_w):
            # 计算当前窗口在 padded 中的位置
            row_start = i * stride
            col_start = j * stride
            row_end = row_start + kh
            col_end = col_start + kw

            # 取出覆盖区域
            region = padded[row_start:row_end, col_start:col_end]

            # 逐元素相乘再求和
            output[i, j] = np.sum(region * kernel)

    return output


# ==================== RGB 图二维卷积 ====================

def conv2d_rgb(image, kernel, padding=0, stride=1):
    """
    对 RGB 图进行二维卷积（对每个通道分别卷积）

    参数：
        image:  3D numpy array，形状 (H, W, 3)
        kernel: 2D numpy array，形状 (kh, kw)
        padding: int
        stride:  int

    返回：
        output: 3D numpy array，形状 (out_h, out_w, 3)
    """
    h, w, c = image.shape
    if c != 3:
        raise ValueError(f"期望 3 通道 RGB 图，实际得到 {c} 通道")

    channels = []
    for ch in range(c):
        result = conv2d_gray(image[:, :, ch], kernel, padding, stride)
        channels.append(result)

    # 沿最后一个维度堆叠，得到 (out_h, out_w, 3)
    return np.stack(channels, axis=-1)


# ==================== 通用卷积函数 ====================

def conv2d(image, kernel, padding=0, stride=1):
    """
    通用二维卷积：自动判断灰度图或 RGB 图

    参数：
        image:  2D (H, W) 或 3D (H, W, 3) numpy array
        kernel: 2D numpy array
        padding: int
        stride:  int

    返回：
        卷积结果，维度与输入一致
    """
    image = np.asarray(image)

    if image.ndim == 2:
        # 灰度图
        return conv2d_gray(image, kernel, padding, stride)
    elif image.ndim == 3:
        # RGB 图
        return conv2d_rgb(image, kernel, padding, stride)
    else:
        raise ValueError(f"不支持的图像维度：{image.ndim}")


# ==================== 卷积核定义 ====================

def mean_kernel(size=3):
    """均值模糊核"""
    return np.ones((size, size), dtype=np.float32) / (size * size)


def gaussian_kernel(size=5, sigma=1.0):
    """高斯模糊核"""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def sharpen_kernel():
    """锐化核"""
    return np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)


def sobel_x_kernel():
    """Sobel-X 垂直边缘检测"""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)


def sobel_y_kernel():
    """Sobel-Y 水平边缘检测"""
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)


# ==================== 结果处理与保存 ====================

def normalize_to_uint8(result):
    """
    把卷积结果归一化到 0~255，转成 uint8，便于保存和显示。
    支持 2D 灰度图和 3D RGB 图。
    """
    result = result.astype(np.float32)

    if result.ndim == 2:
        r_min, r_max = result.min(), result.max()
        if r_max - r_min > 1e-6:
            normalized = (result - r_min) / (r_max - r_min) * 255
        else:
            normalized = np.zeros_like(result)
        return normalized.astype(np.uint8)

    elif result.ndim == 3:
        out = np.zeros_like(result, dtype=np.float32)
        for ch in range(result.shape[2]):
            ch_data = result[:, :, ch]
            r_min, r_max = ch_data.min(), ch_data.max()
            if r_max - r_min > 1e-6:
                out[:, :, ch] = (ch_data - r_min) / (r_max - r_min) * 255
            else:
                out[:, :, ch] = 0
        return out.astype(np.uint8)

    else:
        raise ValueError(f"不支持的维度：{result.ndim}")


def clip_to_uint8(result):
    """
    直接把结果裁剪到 0~255 再转 uint8。
    适用于你希望保留原始亮度关系的情况。
    """
    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== 主流程演示 ====================

def demo(image_path="lane.jpg"):
    """演示灰度图和 RGB 图的卷积"""
    # 1. 读取图片
    img = Image.open(image_path)
    arr = np.array(img)
    print(f"原图：shape={arr.shape}, dtype={arr.dtype}")

    gray = np.array(img.convert("L"))
    print(f"灰度图：shape={gray.shape}, dtype={gray.dtype}")

    # 2. 定义卷积核
    k_mean = mean_kernel(3)
    k_gauss = gaussian_kernel(5, sigma=1.0)
    k_sharp = sharpen_kernel()
    k_sx = sobel_x_kernel()
    k_sy = sobel_y_kernel()

    # 3. 灰度图卷积
    gray_mean = conv2d(gray, k_mean, padding=1, stride=1)
    gray_gauss = conv2d(gray, k_gauss, padding=2, stride=1)
    gray_sharp = conv2d(gray, k_sharp, padding=1, stride=1)
    gray_sx = conv2d(gray, k_sx, padding=1, stride=1)
    gray_sy = conv2d(gray, k_sy, padding=1, stride=1)

    # 边缘合成
    gray_edge = np.sqrt(gray_sx ** 2 + gray_sy ** 2)

    print(f"\n灰度卷积输出形状：{gray_mean.shape}")

    # 4. RGB 图卷积
    rgb_mean = conv2d(arr, k_mean, padding=1, stride=1)
    rgb_sharp = conv2d(arr, k_sharp, padding=1, stride=1)
    print(f"RGB 卷积输出形状：{rgb_mean.shape}")

    # 5. 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("灰度原图")
    axes[0, 1].imshow(normalize_to_uint8(gray_mean), cmap="gray")
    axes[0, 1].set_title("均值模糊")
    axes[0, 2].imshow(normalize_to_uint8(gray_gauss), cmap="gray")
    axes[0, 2].set_title("高斯模糊")
    axes[0, 3].imshow(normalize_to_uint8(gray_sharp), cmap="gray")
    axes[0, 3].set_title("锐化")

    axes[1, 0].imshow(normalize_to_uint8(gray_sx), cmap="gray")
    axes[1, 0].set_title("Sobel-X")
    axes[1, 1].imshow(normalize_to_uint8(gray_sy), cmap="gray")
    axes[1, 1].set_title("Sobel-Y")
    axes[1, 2].imshow(normalize_to_uint8(gray_edge), cmap="gray")
    axes[1, 2].set_title("边缘合成")
    axes[1, 3].imshow(normalize_to_uint8(rgb_sharp))
    axes[1, 3].set_title("RGB 锐化")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("conv_results.png", dpi=150)
    plt.show()

    # 6. 保存结果
    Image.fromarray(normalize_to_uint8(gray_mean)).save("gray_mean.png")
    Image.fromarray(normalize_to_uint8(gray_edge)).save("gray_edge.png")
    Image.fromarray(normalize_to_uint8(rgb_sharp)).save("rgb_sharp.png")

    return {
        "gray_mean": gray_mean,
        "gray_gauss": gray_gauss,
        "gray_sharp": gray_sharp,
        "gray_edge": gray_edge,
        "rgb_sharp": rgb_sharp,
    }


if __name__ == "__main__":
    demo("lane.jpg")