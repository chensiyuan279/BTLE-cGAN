# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.utils.data
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # use non-interactive backend, or error
import matplotlib.pyplot as plt
import metrics
from utils import save_pre_result, save_pre_result1
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
model_path = r"C:\Users\chensiyuan\Desktop\11.6\cgan_ours\bhjcgan_jiamokuai\TITLE+ours_gan_epoch193model.pth"


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


image1_path = r"D:\japan\huairou\huairou2506_3232"
image2_path = r"D:\japan\huairou\huairou2508_3232"
# label_path = r"/root/autodl-tmp/data/test/label_true_test"
# result_path = r"/root/autodl-tmp/data/test/predict"

transform = transforms.Compose([transforms.Resize(image_size),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.1769, 0.2315, 0.1690],[0.1148, 0.1221, 0.1231])
                                # transforms.Normalize([0.1435, 0.1916, 0.1465],[0.0997, 0.1155, 0.0984])
                                # transforms.Normalize([0.1791, 0.2203, 0.1565],[0.0934, 0.1112, 0.0964])
# transforms.Normalize([0.0749, 0.1005, 0.0764],[0.1148, 0.1266, 0.1120])
# transforms.Normalize([0.0764, 0.1005, 0.0749],[0.1120, 0.1266, 0.1148])###门头沟2024——2025年的
# transforms.Normalize([0.0945, 0.1190, 0.1002],[0.1301, 0.1420, 0.1432])
# transforms.Normalize([0.1002, 0.1190, 0.0945],[0.1432, 0.1420, 0.1301])   ##pinggu
# transforms.Normalize([0.0689, 0.0965, 0.0690],[0.1137, 0.1317, 0.1156])
# transforms.Normalize([0.0690, 0.1190, 0.0689],[0.1156, 0.1420, 0.1137])##  miyun
# transforms.Normalize([0.0557, 0.0742, 0.0542],[0.0989, 0.1120, 0.1011])
# transforms.Normalize([0.0542, 0.0742, 0.0557],[0.1011, 0.1120, 0.0989])##  huairou
# transforms.Normalize([0.0839, 0.1027, 0.0890],[0.1216, 0.1285, 0.1321])
# transforms.Normalize([0.0890, 0.1027, 0.0839],[0.1321, 0.1285, 0.1216])##  fangshan
# transforms.Normalize([0.0830, 0.1027, 0.0830],[0.1247, 0.1319, 0.1318])
# transforms.Normalize([0.0830, 0.1027, 0.0830],[0.1318, 0.1319, 0.1247])##  changping
# transforms.Normalize([0.0589, 0.0783, 0.0579],[0.1011, 0.1154, 0.1068])
# transforms.Normalize([0.0579, 0.0783, 0.0589],[0.1068, 0.1154, 0.1011])##  yanqing
# transforms.Normalize([0.0817, 0.1015, 0.0711],[0.1135, 0.1264, 0.1080])
# transforms.Normalize([0.0711, 0.1015, 0.0817],[0.1080, 0.1264, 0.1135])##  门头沟2025.06——08
# transforms.Normalize([0.0653, 0.0943, 0.0653],[0.1100, 0.1304, 0.1138])
# transforms.Normalize([0.0653, 0.0943, 0.0653],[0.1138, 0.1304, 0.1100])##  密云06——08
transforms.Normalize([0.0549, 0.0740, 0.0498],[0.0985, 0.1122, 0.0985])
# transforms.Normalize([0.0498, 0.0740, 0.0549],[0.0985, 0.1122, 0.0985])##  怀柔06——08
# transforms.Normalize([0.0844, 0.1124, 0.0907],[0.1243, 0.1373, 0.1379])
# transforms.Normalize([0.0907, 0.1124, 0.0844],[0.1379, 0.1373, 0.1243])##  平谷06——08
# transforms.Normalize([0.0916, 0.1081, 0.0913],[0.1242, 0.1315, 0.1317])
# transforms.Normalize([0.0913, 0.1081, 0.0916],[0.1317, 0.1315, 0.1242])##  房山06——08
# transforms.Normalize([0.0719, 0.1015, 0.0770],[0.1203, 0.1303, 0.1293])
# transforms.Normalize([0.0770, 0.1015, 0.0719],[0.1293, 0.1303, 0.1203])##  昌平06——08
# transforms.Normalize([0.0599, 0.0821, 0.0566],[0.1014, 0.1191, 0.1052])
# transforms.Normalize([0.0566, 0.0821, 0.0599],[0.1052, 0.1191, 0.1014])##  延庆06——08
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
        # label_files = sorted([os.path.join(label_path, f) for f in os.listdir(label_path)])

        # 确保三个文件夹中的文件数量一致
        assert len(image1_files) == len(image2_files), "三个文件夹中的文件数量不匹配"

        # 初始化 imgs 列表
        imgs = []

        # 将文件路径按顺序组合并添加到 imgs 列表中
        for img1, img2, in zip(image1_files, image2_files,):
            imgs.append((img1, img2,))

        self.imgs = imgs
        self.transform = transform
        # self.label_transform = label_transform  # 标签的图像转换操作

    # 定义返回值
    def __getitem__(self, index):
        fn, label_img, = self.imgs[index]  # fn: 带水印图像路径; label_img: 原始图像路径; label_mask: 标签图像路径

        # 加载带水印图像和原始图像，并转换为RGB模式
        img = Image.open(fn).convert('RGB')
        tar = Image.open(label_img).convert('RGB')

        # # 加载标签图像，转换为单通道模式（L模式表示灰度图像）
        # label = Image.open(label_mask).convert('L')

        # 应用图像转换
        if self.transform is not None:
            img = self.transform(img)
            tar = self.transform(tar)

        # # 标签图像的转换
        # if self.label_transform is not None:
        #     label = self.label_transform(label)

        # 返回路径和图像数据
        return fn, label_img, img, tar, # 返回路径和图像数据

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
        out = self.BTLEcGAN(img1, img2)
        return out


G_net = Generator().to(device)
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(b1, b2))



checkpoint = torch.load(model_path)
new_checkpoint = {}
G_net.load_state_dict(checkpoint['model'])

epoch = 0
G_net.eval()



# # 预测并保存图像
# def predict_and_save(model, dataloader, save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#         print(f'Created directory: {save_dir}')
#
#     model.eval()
#     with torch.no_grad():
#         for i, (x,y, img1,img2,) in enumerate(dataloader):
#             img1 = img1.to(device)
#             img2 = img2.to(device)
#             file_name = os.path.basename(x[0])  # 如果是元组，取第一个元素
#             # print(file_name)
#             # 生成预测
#             pred = model(img1, img2)
#             pred = pred.cpu()  # 将数据移回CPU
#             # 保存预测图像
#             save_pre_result1(pred, file_name, save_dir)



def predict_and_save(model, dataloader, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Created directory: {save_dir}')

    model.eval()
    with torch.no_grad():
        for i, (x, y, img1, img2) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            file_name = os.path.basename(x[0])  # 假设 x 是文件路径列表，取第一个元素作为文件名

            # 生成预测
            pred = model(img1, img2)

            # 如果 pred 是一个元组，提取出预测结果（假设是第一个元素）
            pred = pred[0] if isinstance(pred, tuple) else pred

            # 将数据移回CPU
            pred = pred.cpu()

            # 保存预测图像
            save_pre_result1(pred, file_name, save_dir)


# 运行预测并保存图像
save_dir = r"D:\japan\huairou2\huairou2_2506_08"# 预测图像的保存路径
predict_and_save(G_net, dataloader, save_dir)


