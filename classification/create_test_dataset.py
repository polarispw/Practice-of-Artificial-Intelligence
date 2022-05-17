import os
import cv2
import albumentations as A
from tqdm import tqdm

trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            A.GaussNoise(),    # 将高斯噪声应用于输入图像。
        ], p=0.5),   # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.3),   # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.2),    # 中值滤波
            A.Blur(blur_limit=3, p=0.3),   # 使用随机大小的内核模糊输入图像。
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.4),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.5),   # 随机明亮对比度
    ])

for i in range(4):
    data_path = "datasets"
    save_path = "test_dataset"
    data_path = os.path.join(data_path, str(i))
    save_path = os.path.join(save_path, str(i))
    if os.path.exists(data_path):
        for img_name in tqdm(os.listdir(data_path)):
            img_path = os.path.join(data_path, img_name)
            img = cv2.imread(img_path)
            image = trans(image=img)
            img = image["image"]
            cv2.imwrite(os.path.join(save_path, img_name), img)
    else:
        raise ValueError("Path can not be found")
