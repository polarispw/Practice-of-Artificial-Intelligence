import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from my_dataset import Dataset

from torch.utils.data import DataLoader


def main():
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['1', '2', '3']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    x_train_dir = "Heart Data/Image_DCM/png/Image/01"
    y_train_dir = "Heart Data/Image_DCM/png/Label/01"

    # create segmentation model with pretrained encoder
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    # train model for 40 epochs

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5E-5)
    parser.add_argument('--lrf', type=float, default=1E-3)
    parser.add_argument('--optimizer', type=str, default='Adam', help='choose from SGD and Adam')
    parser.add_argument('--scheduler', type=str, default='', help='write your lr schedule keywords')
    parser.add_argument('--augmentation', type=str, default='', help='interpretation')

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="datasets_1000")

    # load model weights
    parser.add_argument('--weights', type=str, default='weights/best_weight_0523-0135.pth', help='initial weights path')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)