import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from data_process import get_training_augmentation, get_validation_augmentation, get_preprocessing
from my_dataset import Dataset, generate_path_list

from torch.utils.data import DataLoader


def main(args):
    best_model = torch.load('./best_model.pth')

    encoder = args.encoder
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=3,
        activation='sigmoid',
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder)

    # create test dataset
    valid_img_list, valid_lab_list = generate_path_list(args.data_path, args.mode)
    test_dataset = Dataset(
        valid_img_list,
        valid_lab_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataloader = DataLoader(test_dataset)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=args.device,
    )

    logs = test_epoch.run(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="seg", help='seg or cls')
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="test")

    # load model weights
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder backbone')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)