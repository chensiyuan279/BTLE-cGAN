# -*- coding: utf-8 -*-
import functools

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.utils.data
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # use non-interactive backend, or error
import matplotlib.pyplot as plt
import metrics
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
from utils import get_logger
from koutu import segment_image, generate_values
from torchvision import models
from collections import namedtuple

# -------- set parameters -------- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
n_epochs = 300

# Optimizers
lr = 0.0001
b1 = 0.5
b2 = 0.999

# number of images used for training
N = 9936

# loss weights
# langmuda = 1
# alpha = 0.01 # perceptual loss
# beta = 10 # l1 loss


langmuda = 1
alpha = 0.001 # perceptual loss
beta = 1 # l1 loss

# langmuda = 1
# alpha = 0 # L1 loss
# beta = 0 # perceptual loss

# number of threads for loading dataset
workers = 8

# whether to reorder the dataset
shuffle = True

image_size = 256
img_shape = (image_size, image_size, 3)


image1_path = r"/root/autodl-tmp/data/train/2022true_train_zq"
image2_path = r"/root/autodl-tmp/data/train/2023true_train_zq"
label_path = r"/root/autodl-tmp/data/train/label_true_train_zq"
result_path = r"/root/autodl-tmp/data/train/results1"
image1_path_val = r"/root/autodl-tmp/data/val/2022val"
image2_path_val = r"/root/autodl-tmp/data/val/2023val"
label_path_val = r"/root/autodl-tmp/data/val/label_val"

# pretrained_weights = r"/root/autodl-tmp/data/train/results6/Gnet_274.pth"
# pretrained_weightsd = r"/root/autodl-tmp/data/train/results6/Dnet_274.pth"
# pretrained_weights = r"/root/autodl-tmp/dataset/train/BTLEcGAN_LEVIRCD_batch=16_lr=0.0001_epoch197model.pth"

# define transformers
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.1769, 0.2315, 0.1690],
                                                     [0.1148, 0.1221, 0.1231])])
label_transform = transforms.Compose([transforms.Resize(image_size),
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
        img1_path, img2_path, label_path = self.imgs[index]  # fn: 带水印图像路径; label_img: 原始图像路径; label_mask: 标签图像路径

        # 加载带水印图像和原始图像，并转换为RGB模式
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # 加载标签图像，转换为单通道模式（L模式表示灰度图像）
        label = Image.open(label_path).convert('L')

        # 应用图像转换
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 标签图像的转换
        if self.label_transform is not None:
            label = self.label_transform(label)

        # 返回路径和图像数据
        return img1_path, img2_path, label_path, img1, img2, label  # 返回路径和图像数据

    def __len__(self):
        return len(self.imgs)


class MyDataset_val(Dataset):
    def __init__(self, transform=None, label_transform=None):
        # 获取每个文件夹中的文件列表
        image1_files_val = sorted([os.path.join(image1_path_val, f) for f in os.listdir(image1_path_val)])
        image2_files_val = sorted([os.path.join(image2_path_val, f) for f in os.listdir(image2_path_val)])
        label_files_val = sorted([os.path.join(label_path_val, f) for f in os.listdir(label_path_val)])

        # 确保三个文件夹中的文件数量一致
        assert len(image1_files_val) == len(image2_files_val) == len(label_files_val), "三个文件夹中的文件数量不匹配"

        # 初始化 imgs 列表
        imgs_val = []
        # 将文件路径按顺序组合并添加到 imgs 列表中
        for img1, img2, label in zip(image1_files_val, image2_files_val, label_files_val):
            imgs_val.append((img1, img2, label))
        self.imgs_val = imgs_val
        self.transform = transform
        self.label_transform = label_transform  # 标签的图像转换操作
    # 定义返回值
    def __getitem__(self, index):
        img1_path_val, img2_path_val, label_path_val = self.imgs_val[index]  # fn: 带水印图像路径; label_img: 原始图像路径; label_mask: 标签图像路径

        # 加载带水印图像和原始图像，并转换为RGB模式
        img1_val = Image.open(img1_path_val).convert('RGB')
        img2_val = Image.open(img2_path_val).convert('RGB')

        # 加载标签图像，转换为单通道模式（L模式表示灰度图像）
        label_val = Image.open(label_path_val).convert('L')

        # 应用图像转换
        if self.transform is not None:
            img1_val = self.transform(img1_val)
            img2_val = self.transform(img2_val)

        # 标签图像的转换
        if self.label_transform is not None:
            label_val = self.label_transform(label_val)

        # 返回路径和图像数据
        return img1_path_val, img2_path_val, label_path_val, img1_val, img2_val, label_val  # 返回路径和图像数据

    def __len__(self):
        return len(self.imgs_val)


# -------- define dataloader -------- #
train_set = MyDataset(transform, label_transform)
val_set = MyDataset_val(transform, label_transform)
dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)


# -------- define transform converter -------- #
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


# -------- define networks -------- #
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

        # # 如果提供了预训练权重路径，则加载权重
        # if pretrained_weights:
        #     checkpoint = torch.load(pretrained_weights)
        #     # 尝试加载权重到 BTLEcGAN
        #     try:
        #         self.BTLEcGAN.load_state_dict(checkpoint)
        #         print("成功加载预训练权重。")
        #     except RuntimeError as e:
        #         print(f"加载权重时出现错误：{e}")
        #         # 使用严格模式为 False 以忽略不匹配的键
        #         self.BTLEcGAN.load_state_dict(checkpoint, strict=False)
        #         print("部分权重已加载，缺失部分被忽略。")

        # if pretrained_weights:
        #     checkpoint = torch.load(pretrained_weights)
        #     state_dict = checkpoint.get('model', checkpoint)
        #
        #     # 移除前缀 'BTLEcGAN.' 以匹配模型中的参数名称
        #     new_state_dict = {k.replace("BTLEcGAN.", ""): v for k, v in state_dict.items()}
        #
        #     missing_keys, unexpected_keys = self.BTLEcGAN.load_state_dict(new_state_dict, strict=False)
        #     if missing_keys:
        #         print(f"BTLEcGAN - 缺少的键：{missing_keys}")
        #     if unexpected_keys:
        #         print(f"BTLEcGAN - 意外的键：{unexpected_keys}")
        #     print("生成器部分或全部预训练权重成功加载。")

    def forward(self, img1, img2):
        # 使用 BTLEcGAN 的前向传播作为 Generator 的输出
        out = self.BTLEcGAN(img1, img2)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(4, 64, 3, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.3)  # 添加 Dropout
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.5, inplace=True)
        self.norm2 = nn.InstanceNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.3)  # 添加 Dropout
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.5, inplace=True)
        self.norm3 = nn.InstanceNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.3)  # 添加 Dropout
        self.conv3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        # # 如果提供了预训练权重路径，则加载权重
        # if pretrained_weightsd:
        #     checkpointd = torch.load(pretrained_weightsd)
        #     state_dict = checkpointd.get('model', checkpointd)  # 获取模型权重
        #
        #     model_dict = self.state_dict()  # 当前模型的状态字典
        #     # 过滤预训练模型中存在于当前模型的键
        #     filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        #     model_dict.update(filtered_state_dict)
        #
        #     # 加载过滤后的权重
        #     missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
        #     if missing_keys:
        #         print(f"Discriminator - 缺少的键：{missing_keys}")
        #     if unexpected_keys:
        #         print(f"Discriminator - 意外的键：{unexpected_keys}")
        #     print("判别器部分或全部预训练权重成功加载。")

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.dropout1(x)  # 应用 Dropout
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm2(x)
        x = self.dropout2(x)  # 应用 Dropout
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm3(x)
        x = self.dropout3(x)  # 应用 Dropout
        x = self.conv3(x)
        # x = torch.nn.Sigmoid()(x.view(x.size()[0], -1).mean(1))
        x = torch.nn.Sigmoid()(x.view(x.size()[0], -1).mean(1))  # 去掉 Sigmoid，仅返回原始 logits
        return x



# -------- define perceptual loss ---------- #
# change this if your image size is not 256*256
# whc stands for width, height and number of channels of the output of relu2_2
whc = 128*128*128

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # model = models.vgg16(pretrained=False)
        model = models.vgg16(weights=None)
        pre = torch.load('vgg16-397923af.pth')
        model.load_state_dict(pre)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        # Slice 1 -> layers 1-4 of VGG
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        # Slice 2 -> layers 4-9 of VGG
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        # Slice 3 -> layers 9-16 of VGG
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        # Slice 4 -> layers 16-23 of VGG
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.slice1(x)
        relu1_2 = out  # Snapshot output of relu1_2
        out = self.slice2(relu1_2)
        relu2_2 = out
        out = self.slice3(relu2_2)
        relu3_3 = out
        out = self.slice4(relu3_3)
        relu4_3 = out

        output_tuple = namedtuple("VGGOutputs", ['relu1_1', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = output_tuple(relu1_2, relu2_2, relu3_3, relu4_3)
        return out

vggmodel = VGG16().to(device)

def VGGls(gxs, ys):
    gx = gxs
    y = ys
    # #  确保输入为3通道（适配VGG的输入要求）
    # if gx.shape[1] == 1:  # 检查通道数是否为1
    #     gx = gx.repeat(1, 3, 1, 1)  # 将单通道复制到3个通道
    # if y.shape[1] == 1:
    #     y = y.repeat(1, 3, 1, 1)
    features_gx = vggmodel.forward(gx)
    features_y = vggmodel.forward(y)
    t = features_gx.relu3_3-features_y.relu3_3
    t = torch.norm(t)
    pl = (t * t / whc) / batch_size
    return pl


# 1. 位置损失（L1 Loss）
def pixel_loss(fake, real):
    return F.l1_loss(fake, real)


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-8):
        """
        Tversky Loss based on the paper: https://arxiv.org/pdf/1706.05721.pdf
        :param alpha: weight for false positives (default is 0.3)
        :param beta: weight for false negatives (default is 0.7)
        :param smooth: small constant to prevent division by zero (default is 1e-6)
        """
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  # 控制假阳性的重要性
        self.beta = beta    # 控制假阴性的重要性

    def forward(self, fake, real):
        # 计算真阳性、假阳性和假阴性
        intersection = (fake * real).sum()
        false_positive = ((1 - real) * fake).sum()
        false_negative = ((1 - fake) * real).sum()

        # 计算 Tversky Loss
        tversky_loss = 1 - (intersection + self.smooth) / (intersection + self.alpha * false_positive + self.beta * false_negative + self.smooth)
        return tversky_loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.pixel_loss = pixel_loss
        self.TverskyLoss = TverskyLoss(smooth=1e-6)
        # self.edge_loss = EdgeLoss(smooth=1e-6)
        self.alpha = alpha
        self.beta = beta
        # self.gamma = gamma

    def forward(self, fake, real):
        # pl = torch.clamp(self.pixel_loss(fake, real), min=0, max=1)  # 控制在合理范围内
        dl = torch.clamp(self.TverskyLoss(fake, real), min=0, max=1)
        # el = torch.clamp(self.edge_loss(fake, real), min=0, max=1)

        # combined_loss = self.alpha * pl + self.beta * dl
        combined_loss = self.beta * dl
        # combined_loss = self.alpha * pl
        return combined_loss

class STEThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        # 前向传播时二值化，但保存原始概率图以供反向传播使用
        ctx.save_for_backward(input)
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时直接将梯度传递给原始概率图 `input`
        (input,) = ctx.saved_tensors
        return grad_output, None

# 使用该函数来进行“二值化”
def ste_threshold(input, threshold=0.5):
    return STEThresholdFunction.apply(input, threshold)


# # 定义 Hinge Loss 函数
# def hinge_loss_discriminator(real_output, fake_output):
#     """
#     real_output: 判别器对真实样本的输出
#     fake_output: 判别器对生成样本的输出
#     """
#     real_loss = torch.mean(torch.relu(1.0 - real_output))  # 对真实样本
#     fake_loss = torch.mean(torch.relu(1.0 + fake_output))  # 对生成样本
#     return real_loss + fake_loss
#
# def hinge_loss_generator(fake_output):
#     """
#     fake_output: 判别器对生成样本的输出
#     """
#     return -torch.mean(fake_output)  # 生成器希望判别器的输出为正

class MSEGANLoss(nn.Module):
    """Define a GAN loss class specifically using MSELoss."""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(MSEGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, is_real):
        """Ensure target tensor is on the same device as the prediction tensor."""
        if is_real:
            return self.real_label.expand_as(prediction).to(prediction.device)
        else:
            return self.fake_label.expand_as(prediction).to(prediction.device)

    def forward(self, prediction, is_real):
        target_tensor = self.get_target_tensor(prediction, is_real)
        return self.loss(prediction, target_tensor)


# -------- initialize loss, optimizers and networks -------- #
# Loss function

# adv_loss = torch.nn.BCELoss()
criterion_ce = nn.BCELoss()
# combined_loss_fn = CombinedLoss(alpha=1, beta=1)
per_loss = 0.0
gan_loss = MSEGANLoss(target_real_label=1.0, target_fake_label=0.0)

if not os.path.exists('logs'):
    os.makedirs('logs')
logger = get_logger('logs/' + "TITLE" + '.log')
logger.info('Net: ' + "TITLE")

# Initialize generator and discriminator
G_net = Generator().to(device)
D_net =Discriminator().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.0005)
optimizer_D = torch.optim.Adam(D_net.parameters(), lr=lr, betas=(0.5, b2), weight_decay=0.001)

# # 学习率调度器，使用 ReduceLROnPlateau
# scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', patience=10, factor=0.5, verbose=True)
# scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', patience=10, factor=0.5, verbose=True)
# 使用 CosineAnnealingLR 调度器
scheduler_G = CosineAnnealingLR(optimizer_G, T_max=300, eta_min=1e-5)
scheduler_D = CosineAnnealingLR(optimizer_D, T_max=300, eta_min=1e-5)

# G_net.train()
# D_net.train()


def validate(net, dataloader_val, epoch):
    print('Validating...')
    model = net.eval()
    num = 0
    cm_total_val = np.zeros((2, 2))

    for i, (x, y, z, img1_val, img2_val, label_val) in enumerate(dataloader_val):
        img1_val = img1_val.to(device)
        img2_val = img2_val.to(device)
        label_val = label_val.to(device)
        gen_val = model(img1_val, img2_val)
        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm_val = metrics.ConfusionMatrix(2, gen_val, label_val)
        cm_total_val += cm_val
        num += 1
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total_val)
    return precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total

best_f1 = 0
best_epoch = 0

# -------- start to train the networks -------- #
for epoch in range(0, n_epochs):

    # 打印当前学习率
    current_lr = optimizer_G.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

    print('Epoch {}/{}'.format(epoch, n_epochs))
    print('=' * 10)
    cm_total = np.zeros((2, 2))


    for i, (x, y, z, img1, img2, label) in enumerate(dataloader):
        # 获取当前 batch 的实际大小
        current_batch_size = img1.size(0)  # 或者 img2.size(0)

        # 动态生成 ones 和 zeros
        ones = torch.ones(current_batch_size).to(device)
        zeros = torch.zeros(current_batch_size).to(device)
        # ones = torch.ones(batch_size).to(device)
        # zeros = torch.zeros(batch_size).to(device)

        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = torch.abs(torch.sub(img1, img2))
        label = label.to(device)
        # -------- train D net -------- #
        # if i == 0 or (i + 1) % 50 == 0:
        G_net.train()
        D_net.train()

        optimizer_D.zero_grad()

        gen = G_net(img1, img2).detach().to(device)

        real_x = torch.cat((img3, label), dim=1).to(device)
        fake_x = torch.cat((img3, gen), dim=1).to(device)


        # 计算真实样本的损失
        real_loss = gan_loss(D_net(real_x), is_real=True)
        # 计算生成样本的损失
        fake_loss = gan_loss(D_net(fake_x), is_real=False)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(D_net.parameters(), 1)

        # # 计算并监控梯度范数
        # total_norm = 0
        # for p in D_net.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print(f"Gradient Norm for D_net: {total_norm}")

        optimizer_D.step()

        # -------- train G net -------- #
        G_net.train()
        D_net.eval()
        optimizer_G.zero_grad()
        gen = G_net(img1, img2)
        gen_binary = ste_threshold(gen)


        image_for_label_1_gen = segment_image(img2, gen_binary)
        image_for_label_1_yuan = segment_image(img2, label)

        per_loss = VGGls(image_for_label_1_gen, image_for_label_1_yuan)
        combined_loss = criterion_ce(gen, label)
        fake_x = torch.cat((img3, gen), dim=1)
        adv_loss = gan_loss(D_net(fake_x), is_real=True)

        g_loss = langmuda * adv_loss + alpha * per_loss + beta * combined_loss

        cm = metrics.ConfusionMatrix(2, gen, label)
        cm_total += cm
        precision, recall, f1, iou, kc = metrics.get_score(cm)
        print('Pre:%f, Rec:%f, F1:%f, IoU:%f, KC:%f' % (precision[1], recall[1], f1[1], iou[1], kc))

        g_loss.backward()

        # # 计算并监控梯度范数
        # total_norm = 0
        # for p in G_net.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print(f"Gradient Norm for G_net: {total_norm}")
        #
        # # 假设 optimizer 是优化器，max_norm 是梯度裁剪的最大范数
        # torch.nn.utils.clip_grad_norm_(G_net.parameters(), 1)
        # # 计算并打印裁剪后的梯度范数（总的）
        # total_norm_after_clipping = 0
        # for p in G_net.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm_after_clipping += param_norm.item() ** 2
        # total_norm_after_clipping = total_norm_after_clipping ** (1. / 2)
        #
        # print(f"Gradient Norm for G_net after clipping: {total_norm_after_clipping}")


        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Real Loss: %f] [fake loss: %f] [G loss: %f] [adv loss: %f] [per loss: %f] [combined loss: %f]" % (
        epoch, n_epochs, i, len(dataloader), d_loss.item(), real_loss.item(), fake_loss.item(), g_loss.item(), adv_loss, per_loss,
        combined_loss), )

    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    print('epoch %d - , train Pre:%f, train Rec:%f, train F1:%f, train iou:%f, train kc:%f' % (
        epoch, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'],
        kc_total))
    logger.info(
        'Epoch:[{}]\t train_Pre={:.3f}\t train_Rec={:.3f}\t train_F1={:.3f}\t train_IoU={:.3f}\t train_KC={:.3f}'.format(
            epoch, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'],
            kc_total))

    pre_val, recall_val, f1_val, iou_val, kc_val = validate(G_net, dataloader_val, epoch)

    # 对于 CosineAnnealingLR，不需要传递任何参数
    scheduler_G.step()
    scheduler_D.step()

    if f1_val > best_f1:
        best_f1 = f1_val
        best_epoch = epoch
        ckp_name = "TITLE+ours_gan" + '_epoch{}model.pth'.format(epoch)
        stateG = {'model': G_net.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
        torch.save(stateG, os.path.join(result_path, ckp_name), _use_new_zipfile_serialization=False)

    print('epoch %d - val Pre :%f val Recall: %f val F1_Score: %f  iou: %f  kc: %f' % (epoch, pre_val, recall_val, f1_val, iou_val, kc_val))
    logger.info(
        'Epoch:[{}/{}]\t val_Pre={:.4f}\t val_Rec:{:.4f}\t val_F1={:.4f}\t IoU={:.4f}\t KC={:.4f}\t best_F1:[{:.4f}/{}]\t'.format(
            epoch, n_epochs, pre_val, recall_val, f1_val, iou_val, kc_val, best_f1, best_epoch))

    # output results for every 10 epochs
    if (epoch + 1) % 5 == 0:
        plt_size = 1  # plt_size<batch_size, <5 recomm

        # save parameters of the models
        stateG = {'model': G_net.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': epoch}
        torch.save(stateG, result_path + '/Gnet_' + str(epoch) + ".pth")

        stateD = {'model': D_net.state_dict(), 'optimizer': optimizer_D.state_dict(), 'epoch': epoch}
        torch.save(stateD, result_path + '/Dnet_' + str(epoch) + ".pth")

print("Done! batch_size=" + str(batch_size) + ", N=" + str(N) + ", results saved in folder:" + result_path)