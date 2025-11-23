import logging
import torch
from PIL import Image
import numpy as np
import os


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def to8bits(img):
    result = np.ones([img.shape[0], img.shape[1]], dtype='int')
    result[img == 0] = 0
    result[img == 1] = 255
    return result


def save_pre_result(pre, original_filename, save_path):
    # 将预测结果二值化为 0 或 255
    pre[pre >= 0.5] = 255
    pre[pre < 0.5] = 0

    # 将张量转为 NumPy 数组并去掉多余维度
    outputs = pre.cpu().detach().numpy()  # (batch_size, 1, height, width) 或 (batch_size, height, width)

    # 如果 outputs 的形状是 (batch_size, 1, height, width)，去掉单通道维度
    if outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)  # (batch_size, height, width)

    # 逐张保存每张图片
    batch_size = outputs.shape[0]
    for i in range(batch_size):
        img_array = outputs[i]  # 取出单张图片，形状应为 (height, width)
        img_array = np.uint8(img_array)  # 转换为 uint8 类型

        # 检查形状是否符合 (height, width)
        if img_array.ndim != 2:
            raise ValueError(f"Unexpected shape for img_array: {img_array.shape}, expected (height, width)")

        # 创建并保存图像
        img = Image.fromarray(img_array)
        # img.save(os.path.join(save_path, f"{flag}_{num}_{i}.tif"))

        # 生成保存文件名（保留原始文件名，但更改扩展名为 .tif）
        base_name = os.path.splitext(original_filename)[0]  # 去掉扩展名
        save_filename = f"{base_name}.tif"

        # 保存文件
        img.save(os.path.join(save_path, save_filename))


import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_pre_result_t1(pre, original_filename, save_path):
    # 将张量转为 NumPy 数组并去掉多余维度
    outputs = pre.cpu().detach().numpy()  # (batch_size, 1, height, width) 或 (batch_size, height, width)

    # 如果 outputs 的形状是 (batch_size, 1, height, width)，去掉单通道维度
    if outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)  # (batch_size, height, width)

    # 逐张保存每张热力图
    batch_size = outputs.shape[0]
    for i in range(batch_size):
        img_array = outputs[i]  # 取出单张图片，形状应为 (height, width)

        # 检查形状是否符合 (height, width)
        if img_array.ndim != 2:
            raise ValueError(f"Unexpected shape for img_array: {img_array.shape}, expected (height, width)")

        # # 归一化到 [0, 1] 范围
        # img_min = np.min(img_array)
        # img_max = np.max(img_array)
        # img_array = (img_array - img_min) / (img_max - img_min)  # 归一化到 [0, 1]

        # 创建热力图
        plt.imshow(img_array, cmap='hot')  # 使用热力图颜色映射（例如 'hot', 'jet', 'plasma'）
        plt.axis('off')  # 不显示坐标轴

        # 生成保存文件名（保留原始文件名，但更改扩展名为 .tif）
        base_name = os.path.splitext(original_filename)[0]  # 去掉扩展名
        save_filename = f"{base_name}_{i}.tif"

        # 保存文件
        plt.savefig(os.path.join(save_path, save_filename), bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭当前图像，防止内存泄漏



def save_pre_result1(pre, original_filename, save_path):
    """
    保存预测结果，将输出的预测结果保存为与原始图片文件名相同的文件名。

    Args:
        pre (torch.Tensor): 预测结果张量。
        original_filename (str): 原始图片文件名，例如 "example.png"。
        save_path (str): 保存路径。
    """
    # 将预测结果处理为二值图像
    pre[pre >= 0.5] = 255
    pre[pre < 0.5] = 0
    outputs = torch.squeeze(pre).cpu().detach().numpy()
    outputs = Image.fromarray(np.uint8(outputs))

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 生成保存文件名（保留原始文件名，但更改扩展名为 .tif）
    base_name = os.path.splitext(original_filename)[0]  # 去掉扩展名
    save_filename = f"{base_name}.tif"

    # 保存文件
    outputs.save(os.path.join(save_path, save_filename))
