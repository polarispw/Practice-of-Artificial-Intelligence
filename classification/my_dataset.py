from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

def scaleRadius(img,scale):
    x = img[int(img.shape[0]/2),:,:].sum(1) # 图像中间1行的像素的3个通道求和。输出（width*1）
    r = (x>x.mean()/10).sum()/2 # x均值/10的像素是为眼球，计算半径
    s = scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def pre_process(img):
    scale = 300
    a = scaleRadius(img, scale)
    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30),-4, 128)
    # remove out er 10%
    b = np.zeros(a.shape)
    cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    a = a * b + 128 * (1 - b)
    img = Image.fromarray(np.uint8(a))
    return img


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = Image.open(self.images_path[item])
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        img = cv2.imread(self.images_path[item])
        img = pre_process(img)
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.images_path[item]

    @staticmethod
    def collate_fn(batch):
        images, labels, names = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, names
