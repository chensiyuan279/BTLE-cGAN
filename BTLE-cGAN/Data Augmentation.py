###   新的数据增强

import os
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as F
import random

class JointTransformForChangeDetection:
    def __init__(self, resize, rotate_degrees, h_flip_p, v_flip_p, translate, scale, brightness_factor,
                 contrast_factor, saturation_factor, noise_factor=0.01, blur_radius=0.02, shear_factor=0):
        self.resize = resize
        self.rotate_degrees = rotate_degrees
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.translate = translate
        self.scale = scale
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.noise_factor = noise_factor
        self.blur_radius = blur_radius
        self.shear_factor = shear_factor

        # 图像归一化
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1769, 0.2315, 0.1690], std=[0.1148, 0.1221, 0.1231])
        ])

        # 掩膜转换为张量
        self.transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])

    def add_noise(self, image):
        """添加高斯噪声"""
        # 确保输入图像是 PyTorch Tensor 格式
        if isinstance(image, Image.Image):  # 如果是 PIL 图像，先转换为 Tensor
            image = transforms.ToTensor()(image)

        # 获取图像的通道数、高度和宽度
        c, h, w = image.size()  # 这里是 PyTorch Tensor 格式，使用 .size() 获取 (C, H, W)

        # 生成与图像相同形状的噪声
        noise = torch.randn(c, h, w) * self.noise_factor  # 为每个通道生成噪声
        noisy_image = torch.clamp(image + noise, 0, 1)  # 将噪声加到图像上，并确保像素值在[0, 1]范围内
        return noisy_image

    def apply_blur(self, image):
        """添加模糊效果"""
        # 如果是 PyTorch Tensor 格式，转换为 PIL 图像
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # 进行高斯模糊
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # 如果需要返回 PyTorch Tensor，转换回去
        return transforms.ToTensor()(blurred_image)

    def apply_shear(self, image, mask):
        """随机仿射剪切"""
        shear_x = random.uniform(-self.shear_factor, self.shear_factor)
        shear_y = random.uniform(-self.shear_factor, self.shear_factor)

        # 对图像进行仿射变换
        img1 = F.affine(image, angle=0, translate=(0, 0), scale=1, shear=(shear_x, shear_y))

        # 对掩膜进行仿射变换（使用 Image.NEAREST 插值）
        mask = F.affine(mask, angle=0, translate=(0, 0), scale=1, shear=(shear_x, shear_y))  # 省略 resample 参数

        return img1, mask

    def __call__(self, img1, img2, mask):
        # 调整图像和掩膜大小
        img1 = transforms.functional.resize(img1, self.resize)
        img2 = transforms.functional.resize(img2, self.resize)
        mask = transforms.functional.resize(mask, self.resize, interpolation=transforms.InterpolationMode.NEAREST)

        # 生成共享的几何增强参数
        affine_params = transforms.RandomAffine.get_params(
            degrees=(-self.rotate_degrees, self.rotate_degrees),
            translate=(self.translate, self.translate),
            scale_ranges=(1.0 - self.scale, 1.0 + self.scale),
            shears=(0, 0),
            img_size=img1.size
        )

        # 水平和垂直翻转
        h_flip = torch.rand(1) < self.h_flip_p
        v_flip = torch.rand(1) < self.v_flip_p

        if h_flip:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
            mask = F.hflip(mask)

        if v_flip:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)
            mask = F.vflip(mask)

        # 仿射变换
        img1 = F.affine(img1, *affine_params, interpolation=transforms.InterpolationMode.BILINEAR, fill=(0, 0, 0))
        img2 = F.affine(img2, *affine_params, interpolation=transforms.InterpolationMode.BILINEAR, fill=(0, 0, 0))
        mask = F.affine(mask, *affine_params, interpolation=transforms.InterpolationMode.NEAREST, fill=0)

        # 颜色增强
        brightness = torch.empty(1).uniform_(1 - self.brightness_factor, 1 + self.brightness_factor).item()
        contrast = torch.empty(1).uniform_(1 - self.contrast_factor, 1 + self.contrast_factor).item()
        saturation = torch.empty(1).uniform_(1 - self.saturation_factor, 1 + self.saturation_factor).item()

        img1 = F.adjust_brightness(img1, brightness)
        img1 = F.adjust_contrast(img1, contrast)
        img1 = F.adjust_saturation(img1, saturation)

        img2 = F.adjust_brightness(img2, brightness)
        img2 = F.adjust_contrast(img2, contrast)
        img2 = F.adjust_saturation(img2, saturation)

        # 添加噪声和模糊
        img1 = self.add_noise(img1)
        img2 = self.add_noise(img2)
        img1 = self.apply_blur(img1)
        img2 = self.apply_blur(img2)

        # 应用剪切
        img1, mask = self.apply_shear(img1, mask)
        img2, _ = self.apply_shear(img2, mask)

        # 确保图像为 PIL 格式后再归一化
        img1 = transforms.ToPILImage()(img1) if isinstance(img1, torch.Tensor) else img1
        img2 = transforms.ToPILImage()(img2) if isinstance(img2, torch.Tensor) else img2

        # 归一化
        img1 = self.normalize_transform(img1)
        img2 = self.normalize_transform(img2)

        # 转换掩膜为张量
        mask = self.transform_mask(mask)

        return img1, img2, mask


# 反归一化函数，将像素值恢复到原始范围
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

# 保存图像时添加反归一化
def save_image(tensor, path, mean, std):
    try:
        # 反归一化
        tensor = denormalize(tensor, mean, std)
        # 确保像素值在 [0, 1] 范围内
        tensor = torch.clamp(tensor, 0, 1)
        # 转换为 PIL 图像
        image = transforms.ToPILImage()(tensor)
        image.save(path)
    except Exception as e:
        print(f"Error saving image: {e}")

# 数据增强和保存的函数
def augment_and_save_change_detection(image1_dir, image2_dir, mask_dir,
                                      output_image1_dir, output_image2_dir, output_mask_dir, transform, num_augments=6):
    # 确保保存目录存在
    os.makedirs(output_image1_dir, exist_ok=True)
    os.makedirs(output_image2_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image1_filenames = os.listdir(image1_dir)

    for image1_filename in image1_filenames:
        image2_filename = image1_filename  # 假设两幅图像同名
        mask_filename = image1_filename  # 假设标签与图像同名

        image1_path = os.path.join(image1_dir, image1_filename)
        image2_path = os.path.join(image2_dir, image2_filename)
        mask_path = os.path.join(mask_dir, mask_filename)

        # 打开两幅输入图像和标签
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 对每组图像进行多次增强
        for i in range(num_augments):
            # 执行增强
            augmented_img1, augmented_img2, augmented_mask = transform(img1, img2, mask)

            # 构建增强后的图像和标签保存路径
            new_image1_filename = f"{os.path.splitext(image1_filename)[0]}_aug_{i}.tif"
            new_image2_filename = f"{os.path.splitext(image2_filename)[0]}_aug_{i}.tif"
            new_mask_filename = f"{os.path.splitext(mask_filename)[0]}_aug_{i}.tif"

            # 保存增强后的两幅图像和掩膜，注意保存图像前需要反归一化
            save_image(augmented_img1, os.path.join(output_image1_dir, new_image1_filename),
                       mean=[0.1769, 0.2315, 0.1690], std=[0.1148, 0.1221, 0.1231])
            save_image(augmented_img2, os.path.join(output_image2_dir, new_image2_filename),
                       mean=[0.1769, 0.2315, 0.1690], std=[0.1148, 0.1221, 0.1231])
            save_image(augmented_mask, os.path.join(output_mask_dir, new_mask_filename),
                       mean=[0], std=[1])  # 掩膜无需反归一化



# 示例路径
image1_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\2022true-train'
image2_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\2023true-train'
mask_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\label_true_train'

output_image1_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\2022true_train_zq'
output_image2_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\2023true_train_zq'
output_mask_dir = r'C:\Users\chensiyuan\Desktop\9_10\xinxinxin\label_true_train_zq'

# 定义数据增强
transform = JointTransformForChangeDetection(
    resize=(256, 256),
    rotate_degrees=15,
    h_flip_p=0.5,
    v_flip_p=0.01,
    translate=0.05,
    scale=0.05,
    brightness_factor=0.2,
    contrast_factor=0.2,
    saturation_factor=0.1
)

# 执行数据增强并保存
augment_and_save_change_detection(image1_dir, image2_dir, mask_dir,
                                  output_image1_dir, output_image2_dir, output_mask_dir, transform)
