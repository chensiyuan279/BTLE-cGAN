import torch

def segment_image(image_tensor, label_tensor):
    """Segment an image tensor based on the label tensor and return the segmented parts.

    Args:
        image_tensor: Tensor of shape (batch_size, channels, height, width).
        label_tensor: Tensor of shape (batch_size, 1, height, width), with 1 representing the pixels to be kept.

    Returns:
        image_for_label_1: Tensor with the pixels where label is 1, other pixels set to 0.
    """
    # 创建浮点掩码以保留计算图
    # mask = (label_tensor == 1).float()  # shape: [batch_size, 1, height, width]
    mask = label_tensor.float().requires_grad_(True)  # 确保它是浮点数并且需要梯度

    # 扩展掩码的维度，使其可以与 image_tensor 相乘
    mask_expanded = mask.expand_as(image_tensor)  # shape: [batch_size, channels, height, width]

    # 将掩码应用到 image_tensor 上，确保计算图包含 gen_binary
    image_for_label_1 = image_tensor * mask_expanded

    return image_for_label_1


# 生成随机值的函数
def generate_values(labels):
    # 标签为 1 时生成 0.9-1 的随机数，标签为 0 时生成 0-0.1 的随机数
    values = torch.where(labels == 1,
                         torch.rand_like(labels, dtype=torch.float) * 0.05 + 0.95, torch.rand_like(labels, dtype=torch.float) * 0.05)        # 生成 0-0.1
    return values