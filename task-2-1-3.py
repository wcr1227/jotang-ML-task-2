# 不好意思，这段写重复了，请忽略
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ==================== 手写二维卷积 ====================

def conv2d_gray(image, kernel, padding=0, stride=1):
    """灰度图二维卷积（手写核心）"""
    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    h, w = image.shape
    kh, kw = kernel.shape

    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"输出尺寸不合法：{out_h}×{out_w}")

    if padding > 0:
        padded = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded = image

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = padded[
                i * stride : i * stride + kh,
                j * stride : j * stride + kw
            ]
            output[i, j] = np.sum(region * kernel)

    return output


def conv2d_rgb(image, kernel, padding=0, stride=1):
    """RGB 图二维卷积：对每个通道分别卷积"""
    h, w, c = image.shape
    channels = []
    for ch in range(c):
        result = conv2d_gray(image[:, :, ch], kernel, padding, stride)
        channels.append(result)
    return np.stack(channels, axis=-1)


def conv2d(image, kernel, padding=0, stride=1):
    """通用卷积：自动判断灰度或 RGB"""
    image = np.asarray(image)
    if image.ndim == 2:
        return conv2d_gray(image, kernel, padding, stride)
    elif image.ndim == 3:
        return conv2d_rgb(image, kernel, padding, stride)
    else:
        raise ValueError(f"不支持的维度：{image.ndim}")


# ==================== 四种卷积核 ====================

def mean_kernel_3x3():
    """3×3 均值模糊核"""
    return np.ones((3, 3), dtype=np.float32) / 9.0


def gaussian_kernel_5x5(sigma=1.0):
    """
    5×5 高斯模糊核
    高斯函数：G(x, y) = exp(-(x² + y²) / (2σ²))
    以中心为原点，生成 5×5 的权重，再归一化
    """
    size = 5
    ax = np.arange(-size // 2 + 1, size // 2 + 1)  # [-2, -1, 0, 1, 2]
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化，保证总和为 1
    return kernel.astype(np.float32)


def sharpen_kernel():
    """锐化核"""
    return np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)


def sobel_x_kernel():
    """Sobel-X：检测竖直边缘"""
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)


def sobel_y_kernel():
    """Sobel-Y：检测水平边缘"""
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)


# ==================== 数值处理 ====================

def normalize_to_uint8(result):
    """把卷积结果归一化到 0~255，转 uint8 便于显示和保存"""
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


# ==================== 主实验 ====================

def run_experiment(image_path="lane.jpg"):
    """对同一张图片应用四种卷积核并对比"""

    # 1. 读取图片
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    gray = np.array(img.convert("L"))
    print(f"原图：{arr.shape}, 灰度图：{gray.shape}")

    # 2. 定义四种卷积核
    k_mean = mean_kernel_3x3()
    k_gauss = gaussian_kernel_5x5(sigma=1.0)
    k_sharp = sharpen_kernel()
    k_sx = sobel_x_kernel()
    k_sy = sobel_y_kernel()

    print("\n5×5 高斯核：")
    print(np.round(k_gauss, 4))
    print(f"高斯核总和：{k_gauss.sum():.4f}")

    # 3. 对灰度图做卷积（使用 same padding 保持尺寸）
    gray_mean = conv2d(gray, k_mean, padding=1)      # 3×3 → padding=1
    gray_gauss = conv2d(gray, k_gauss, padding=2)    # 5×5 → padding=2
    gray_sharp = conv2d(gray, k_sharp, padding=1)
    gray_sx = conv2d(gray, k_sx, padding=1)
    gray_sy = conv2d(gray, k_sy, padding=1)

    # 4. 边缘合成：sqrt(Sobel_X² + Sobel_Y²)
    gray_edge = np.sqrt(gray_sx ** 2 + gray_sy ** 2)

    print(f"\n所有结果形状：{gray_mean.shape}")

    # 5. 对 RGB 图做均值、高斯、锐化（保留颜色）
    rgb_mean = conv2d(arr, k_mean, padding=1)
    rgb_gauss = conv2d(arr, k_gauss, padding=2)
    rgb_sharp = conv2d(arr, k_sharp, padding=1)

    # 6. 可视化：2 行 5 列
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # 第一行：原图 + 灰度结果
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("原图（灰度）", fontsize=12)

    axes[0, 1].imshow(normalize_to_uint8(gray_mean), cmap="gray")
    axes[0, 1].set_title("均值模糊 3×3", fontsize=12)

    axes[0, 2].imshow(normalize_to_uint8(gray_gauss), cmap="gray")
    axes[0, 2].set_title("高斯模糊 5×5", fontsize=12)

    axes[0, 3].imshow(normalize_to_uint8(gray_sharp), cmap="gray")
    axes[0, 3].set_title("锐化", fontsize=12)

    axes[0, 4].imshow(normalize_to_uint8(gray_edge), cmap="gray")
    axes[0, 4].set_title("边缘检测（合成）", fontsize=12)

    # 第二行：RGB 结果 + Sobel 分解
    axes[1, 0].imshow(normalize_to_uint8(rgb_mean))
    axes[1, 0].set_title("RGB 均值模糊", fontsize=12)

    axes[1, 1].imshow(normalize_to_uint8(rgb_gauss))
    axes[1, 1].set_title("RGB 高斯模糊", fontsize=12)

    axes[1, 2].imshow(normalize_to_uint8(rgb_sharp))
    axes[1, 2].set_title("RGB 锐化", fontsize=12)

    axes[1, 3].imshow(normalize_to_uint8(gray_sx), cmap="gray")
    axes[1, 3].set_title("Sobel-X（竖直边缘）", fontsize=12)

    axes[1, 4].imshow(normalize_to_uint8(gray_sy), cmap="gray")
    axes[1, 4].set_title("Sobel-Y（水平边缘）", fontsize=12)

    for ax in axes.flatten():
        ax.axis("off")

    plt.suptitle("四种卷积核处理同一张图片对比", fontsize=16)
    plt.tight_layout()
    plt.savefig("conv_kernels_comparison.png", dpi=150)
    plt.show()

    # 7. 单独保存每张结果图
    Image.fromarray(normalize_to_uint8(gray_mean)).save("result_mean.png")
    Image.fromarray(normalize_to_uint8(gray_gauss)).save("result_gauss.png")
    Image.fromarray(normalize_to_uint8(gray_sharp)).save("result_sharpen.png")
    Image.fromarray(normalize_to_uint8(gray_edge)).save("result_edge.png")
    Image.fromarray(normalize_to_uint8(gray_sx)).save("result_sobel_x.png")
    Image.fromarray(normalize_to_uint8(gray_sy)).save("result_sobel_y.png")
    Image.fromarray(normalize_to_uint8(rgb_sharp)).save("result_rgb_sharpen.png")

    print("\n已保存：conv_kernels_comparison.png 及 7 张单独结果图")

    return {
        "gray_mean": gray_mean,
        "gray_gauss": gray_gauss,
        "gray_sharp": gray_sharp,
        "gray_sx": gray_sx,
        "gray_sy": gray_sy,
        "gray_edge": gray_edge,
        "rgb_sharp": rgb_sharp,
    }


if __name__ == "__main__":
    run_experiment("lane.jpg")