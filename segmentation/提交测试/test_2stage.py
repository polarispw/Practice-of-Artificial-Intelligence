import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from PIL import Image
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from data_process import get_validation_augmentation, get_preprocessing
from my_dataset import Dataset, generate_path_list
from model import efficientnetv2_s as create_model
from utils import collate


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seg
    model = torch.load(args.seg_weight_path)
    encoder = "resnet18"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder)

    # create test dataset
    valid_img_list = generate_path_list(args.data_path)
    test_dataset = Dataset(
        valid_img_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    test_dataloader = DataLoader(test_dataset)

    # cls
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    cls_model = create_model(num_classes=3).to(device)
    model_weight_path = args.cls_weight_path
    cls_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    cls_model.eval()

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # predict
    pre_folder = ''
    f = True
    for n, i in enumerate(tqdm(test_dataloader)):

        # 分割
        image = i.to(device)
        pr_mask = model.predict(image)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        classes = [85, 170, 255]
        mask = np.zeros((384, 480))
        for l in range(pr_mask.shape[0]):
            layer = np.where((pr_mask[l] > 0) & (mask == 0), classes[l], 0)
            mask += layer
        lab = Image.fromarray(mask).convert('L')
        mask = cv2.resize(mask, (256, 208))

        # 分类
        raw = Image.open(valid_img_list[n])
        raw = raw.resize((480, 384))
        cls_img = Image.blend(raw, lab, 0.3)
        cls_img = cls_img.convert("RGB")
        cls_img = data_transform(cls_img)
        # expand batch dimension
        cls_img = torch.unsqueeze(cls_img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(cls_model(cls_img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cls = torch.argmax(predict).item()
            cls_res = class_indict[str(predict_cls)]

        # 输出
        path = valid_img_list[n]
        name = "label" + path.split('\\')[2].split('e')[1].split('.')[0] + f"-{cls_res}" + ".png"
        save_dir = os.path.join("Label", path.split('\\')[1])
        os.makedirs(save_dir) if not os.path.exists(save_dir) else ...
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, mask)
        if pre_folder != save_dir and f is False:
            collate(pre_folder)
            pre_folder = save_dir
        elif f:
            f = False
            collate(save_dir)
            pre_folder = save_dir
    collate(save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="Image")
    parser.add_argument('--seg_weight-path', type=str, default='best_weight_seg.pth')
    parser.add_argument('--cls_weight-path', type=str, default='best_weight_cls.pth')

    opt = parser.parse_args()

    main(opt)
