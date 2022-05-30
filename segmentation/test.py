import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from data_process import get_validation_augmentation, get_preprocessing
from my_dataset import Dataset, generate_path_list
from utils import ValidEpoch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(2, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()


def main(args):
    best_model = torch.load(args.weight_path)

    encoder = args.encoder
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=3,
        activation='sigmoid',
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder)

    # create test dataset
    _, _, valid_img_list, valid_lab_list = generate_path_list(args.data_path, args.mode)
    test_dataset = Dataset(
        valid_img_list,
        valid_lab_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)

    loss = smp.losses.DiceLoss('multilabel')
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # evaluate model on test set
    test_epoch = ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=args.device,
    )

    logs = test_epoch.run(test_dataloader)

    if args.visual:
        n = np.random.choice(len(test_dataset))
        x_train_dir = ["Heart Data/Image_DCM/png/Image/01/image1.png"]
        y_train_dir = ["Heart Data/Image_DCM/png/Label/01/label1.png"]
        test_dataset_vis = Dataset(x_train_dir, y_train_dir)
        test_dataset_pre = Dataset(x_train_dir, y_train_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
        image_vis, mask = test_dataset_vis[0]
        image, gt_mask = test_dataset_pre[0]

        x_tensor = torch.from_numpy(image).to(args.device).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            gt_mask1=gt_mask[0, :, :],
            gt_mask2=gt_mask[1, :, :],
            gt_mask3=gt_mask[2, :, :],
            pr_image=image.transpose(1, 2, 0),
            pr_mask1=pr_mask[0, :, :],
            pr_mask2=pr_mask[1, :, :],
            pr_mask3=pr_mask[2, :, :]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="seg", help='seg or cls')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--visual', type=bool, default=True)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="Heart Data")
    parser.add_argument('--weight-path', type=str, default='runs/2022_0530-12_38_07/best_weight.pth')

    # load model weights
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder backbone')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)