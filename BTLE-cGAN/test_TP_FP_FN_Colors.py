# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.utils.data
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # use non-interactive backend, or error
import matplotlib.pyplot as plt
import metrics
from utils import save_pre_result, save_pre_result1, save_pre_result_t1
import argparse
import os
import numpy as np
import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.BTLEcGAN import BTLEcGAN
from torchvision import models
from collections import namedtuple

# -------- set parameters -------- #
model_path = r"/root/autodl-tmp/data/train/results1/TITLE+ours_gan_epoch193model.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
n_epochs = 1
N = 206  # number of images for test
workers = 0
shuffle = False
lr = 0.0001
b1 = 0.5
b2 = 0.999
# 图片大小，tensor不允许图像大小不一致，所以还是把每张图都裁成方的
image_size = 256
img_shape = (image_size, image_size, 3)

cm_total = np.zeros((2, 2))



image1_path = r"/root/autodl-tmp/data/test/2022true-test"
image2_path = r"/root/autodl-tmp/data/test/2023true-test"
label_path = r"/root/autodl-tmp/data/test/label_true_test"
result_path = r"/root/autodl-tmp/data/test/predictsanse"
result_path_t1 = r"/root/autodl-tmp/data/test/predictsanse_t1"
result_path_t2 = r"/root/autodl-tmp/data/test/predictsanse_t2"


# image1_path = r"/root/autodl-tmp/data/test1/20233636"
# image2_path = r"/root/autodl-tmp/data/test1/20243636"
# label_path = r"/root/autodl-tmp/data/test1/23_24erzhi3636"
# result_path = r"/root/autodl-tmp/data/test1/predict"

# image1_path = r"/root/autodl-tmp/data/henduanshanmai/henduan20073636.tif"
# image2_path = r"/root/autodl-tmp/data/henduanshanmai/henduan20083636.tif"
# label_path = r"/root/autodl-tmp/data/henduanshanmai/henduanbiaoqianerzhi3636"
# result_path = r"/root/autodl-tmp/data/henduanshanmai/hdsmtestsanse"


# image1_path = r"/root/autodl-tmp/data/test2/20233232"
# image2_path = r"/root/autodl-tmp/data/test2/20243232"
# label_path = r"/root/autodl-tmp/data/test2/23-24labelerzhi3232"
# result_path = r"/root/autodl-tmp/data/test2/23_24testsanse"


import os
import numpy as np
from PIL import Image
import torch

def save_three_color_result(pre, label, original_filename, save_path,
                            threshold=0.5, apply_sigmoid=False,
                            suffix="_TPgreen_FPred_FNblue", ext=".png"):
    """
    生成三色误差图：红=FP，绿=TP，蓝=FN，黑=TN。
    - pre: 模型输出，形状 (B,1,H,W) 或 (B,H,W)，dtype=float（logits或概率）
    - label: 标签，形状 (B,1,H,W) 或 (B,H,W)，0/1 或 float
    - original_filename: 原图文件名（用于生成保存名）
    - save_path: 保存目录
    - threshold: 概率阈值（若 pre 是 logits 且 apply_sigmoid=True，会先做 sigmoid 再阈值）
    - apply_sigmoid: True 表示 pre 为 logits；False 表示 pre 已是概率/0-1
    - suffix/ext: 输出文件名后缀与扩展名（建议 .png 以便 RGB）
    """
    os.makedirs(save_path, exist_ok=True)

    # 不做就地修改，避免影响后续指标计算
    pre_t = pre.detach().clone()
    lab_t = label.detach().clone()

    # 统一到 CPU
    pre_t = pre_t.float().cpu()
    lab_t = lab_t.float().cpu()

    # 可选 sigmoid（logits -> prob）
    if apply_sigmoid:
        pre_t = torch.sigmoid(pre_t)

    # squeeze 到 (B,H,W)
    def squeeze_to_bhw(t):
        if t.dim() == 4 and t.size(1) == 1:
            return t[:, 0, ...]
        elif t.dim() == 3:
            return t
        else:
            raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}; expected (B,1,H,W) or (B,H,W).")
    pre_bhw = squeeze_to_bhw(pre_t)
    lab_bhw = squeeze_to_bhw(lab_t)

    # 二值化到 bool
    pred_bin = (pre_bhw >= threshold)
    # 标签：兼容0/1或float
    label_bin = (lab_bhw > 0.5)

    B, H, W = pred_bin.shape

    base_name = os.path.splitext(os.path.basename(original_filename))[0]

    for i in range(B):
        pred_i = pred_bin[i]
        lab_i  = label_bin[i]

        TP = pred_i & lab_i
        FP = pred_i & (~lab_i)
        FN = (~pred_i) & lab_i
        # TN = ~(TP | FP | FN)  # 若以后想给TN上色，可用这行

        # b = (FP.to(torch.uint8) * 255).numpy()   # 红：误检
        # r = (TP.to(torch.uint8) * 255).numpy()   # 绿：正确检出
        # g = (FN.to(torch.uint8) * 255).numpy()   # 蓝：漏检
        # rgb = np.stack([r, g, b], axis=-1)       # (H,W,3)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # FP = 黄色
        rgb[FP.numpy()] = [0, 255, 0]

        # TP = 浅绿色
        rgb[TP.numpy()] = [255, 255, 0]

        # FN = 红色
        rgb[FN.numpy()] = [255, 0, 0]




        img = Image.fromarray(rgb, mode="RGB")

        out_name = f"{base_name}{suffix}{ext}"
        img.save(os.path.join(save_path, out_name))




transform = transforms.Compose([transforms.Resize(image_size),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.1769, 0.2315, 0.1690],[0.1148, 0.1221, 0.1231])
                                # transforms.Normalize([0.1435, 0.1916, 0.1465],[0.0997, 0.1155, 0.0984])  ##23-24门头沟测试
                                # transforms.Normalize([0.1465, 0.1916, 0.1435],[0.0984, 0.1155, 0.0997])
                                # transforms.Normalize([0.1791, 0.2203, 0.1565],[0.0934, 0.1112, 0.0964])
                                # transforms.Normalize(mean=[0.3930, 0.4178, 0.3425], std=[0.2116, 0.2214, 0.2175])    ###横断山脉
                                # transforms.Normalize(mean=[0.3425, 0.4178, 0.3930], std=[0.2175, 0.2214, 0.2116])    ###横断山脉
# transforms.Normalize(mean=[0.1491, 0.2144, 0.1719], std=[0.0851, 0.0991, 0.0832])##2023-2024mentougou
                                ])
label_transform = transforms.Compose([transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
                                      transforms.ToTensor(), ])

img_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])



# -------- define dataset -------- #
class MyDataset(Dataset):
    def __init__(self, transform=None, label_transform=None):
        # 获取每个文件夹中的文件列表
        image1_files = sorted([os.path.join(image1_path, f) for f in os.listdir(image1_path)])
        image2_files = sorted([os.path.join(image2_path, f) for f in os.listdir(image2_path)])
        label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path)])

        # 确保三个文件夹中的文件数量一致
        assert len(image1_files) == len(image2_files) == len(label_files), "三个文件夹中的文件数量不匹配"

        # 初始化 imgs 列表
        imgs = []

        # 将文件路径按顺序组合并添加到 imgs 列表中
        for img1, img2, label in zip(image1_files, image2_files, label_files):
            imgs.append((img1, img2, label))

        self.imgs = imgs
        self.transform = transform
        self.label_transform = label_transform  # 标签的图像转换操作

    # 定义返回值
    def __getitem__(self, index):
        fn, label_img, label_mask = self.imgs[index]  # fn: 带水印图像路径; label_img: 原始图像路径; label_mask: 标签图像路径

        # 加载带水印图像和原始图像，并转换为RGB模式
        img = Image.open(fn).convert('RGB')
        tar = Image.open(label_img).convert('RGB')

        # 加载标签图像，转换为单通道模式（L模式表示灰度图像）
        label = Image.open(label_mask).convert('L')

        # 应用图像转换
        if self.transform is not None:
            img = self.transform(img)
            tar = self.transform(tar)

        # 标签图像的转换
        if self.label_transform is not None:
            label = self.label_transform(label)

        # 返回路径和图像数据
        return fn, label_img, label_mask, img, tar, label  # 返回路径和图像数据

    def __len__(self):
        return len(self.imgs)


# -------- define dataloader -------- #
train_set = MyDataset(transform, label_transform)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    img_tensor = img_tensor.cpu()
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def label_transform_convert(label_tensor):
    """
    将标签图像张量转换为可视化格式
    参数:
    - label_tensor: 标签的 PyTorch 张量，单通道
    """
    # 将张量移动到 CPU 并去除 batch 维度
    label_tensor = label_tensor.cpu().squeeze(0)  # 去掉 batch 维度（如果有）
    # 将标签张量的范围 [0, 1] 转换为 [0, 255]（通常是二值图像）
    label_tensor = label_tensor * 255
    # 将张量从 (C, H, W) 转换为 (H, W) 的 NumPy 数组
    label_np = label_tensor.numpy().astype('uint8').squeeze()  # 去除单通道维度
    # 转换为 PIL 格式，并设置为灰度模式
    label_img = Image.fromarray(label_np, mode='L')  # 'L' 表示单通道灰度图
    return label_img


# -------- define network -------- #
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        self.BTLEcGAN = BTLEcGAN(in_ch=3, out_ch=1, ratio=0.5)

    def forward(self, img1, img2):
        out, t2, t1 = self.BTLEcGAN(img1, img2)
        return out, t2, t1


G_net = Generator().to(device)
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(b1, b2))
checkpoint = torch.load(model_path)

new_checkpoint = {}
#
# for key in checkpoint.keys():
#     # 添加前缀"BTLE-cGAN."以匹配模型中的键
#     new_key = "BTLE-cGAN." + key
#     new_checkpoint[new_key] = checkpoint[key]

# # 加载修改后的状态字典到模型
# G_net.load_state_dict(new_checkpoint)

G_net.load_state_dict(checkpoint['model'])
# G_net.load_state_dict(checkpoint)
# G_net.load_state_dict(checkpoint['model'], strict=False)
# optimizer_G.load_state_dict(checkpoint['optimizer'])
# epoch = checkpoint['epoch']
epoch = 0
G_net.eval()


##  有标签的测试
def predict(net, dataloader_test):
    print('Testing...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    for i, (x,y, z, img1,img2, label) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        file_name = os.path.basename(x[0])  # 如果是元组，取第一个元素
        pre, t2, t1 = model(img1, img2)
        cm = metrics.ConfusionMatrix(2, pre, label)
        cm_total += cm
        # save_pre_result(pre, file_name, save_path=result_path)

        save_pre_result_t1(t1, file_name, save_path=result_path_t1)
        save_pre_result_t1(t2, file_name, save_path=result_path_t2)



        # 新增：三色误差图（红=FP、绿=TP、蓝=FN）
        save_three_color_result(pre, label, file_name, result_path,
                                threshold=0.5, apply_sigmoid=False)


        num += 1
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total, recall_total, f1_total, iou_total, kc_total

if __name__ == '__main__':
    pre_test, rec_test, f1_test, iou_test, kc_test = predict(G_net, dataloader)
    print('test Pre:(%f,%f) test Recall:(%f,%f) test MeanF1Score:(%f,%f) test IoU:(%f,%f) test KC: %f' % (
        pre_test['precision_0'], pre_test['precision_1'], rec_test['recall_0'], rec_test['recall_1'], f1_test['f1_0'],
        f1_test['f1_1'], iou_test['iou_0'], iou_test['iou_1'], kc_test))



