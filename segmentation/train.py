import torch
import argparse
import numpy as np
import segmentation_models_pytorch as smp
from data_process import get_training_augmentation, get_validation_augmentation, get_preprocessing
from my_dataset import Dataset, generate_path_list

from torch.utils.data import DataLoader


def main(args):

    train_img_list, train_lab_list, valid_img_list, valid_lab_list = generate_path_list(args.data_path, args.mode)

    # create segmentation model with pretrained encoder
    encoder = args.encoder
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=3,
        activation='sigmoid',  # could be None for logits or 'softmax2d' for multiclass segmentation
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder)

    train_dataset = Dataset(
        train_img_list,
        train_lab_list,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        valid_img_list,
        valid_lab_list,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
    ])

    # create epoch runners
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=args.device,
        verbose=True,
    )

    max_score = 0

    for epoch in range(args.epochs):

        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="seg", help='seg or cls')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5E-4)
    parser.add_argument('--optimizer', type=str, default='Adam', help='choose from SGD and Adam')
    parser.add_argument('--scheduler', type=str, default='', help='write your lr schedule keywords')

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="Heart Data")

    # load model weights
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder backbone')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)