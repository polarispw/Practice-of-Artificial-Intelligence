import torch
import os
import argparse
import datetime
import json
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses
from data_process import get_training_augmentation, get_validation_augmentation, get_preprocessing
from my_dataset import Dataset, generate_path_list
from utils import TrainEpoch, ValidEpoch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main(args):
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if os.path.exists("./runs") is False:
        os.makedirs("./runs")
    log_path = "./runs/{}".format(datetime.datetime.now().strftime("%Y_%m%d-%H_%M_%S"))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_writer = SummaryWriter(log_dir=log_path)
    with open(log_path + "/arg_list_epoch[{}].json".format(0), "w") as f:
        f.write(json.dumps(vars(args)))

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

    loss = smp.losses.DiceLoss('multilabel')
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
    ])

    # create epoch runners
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=args.device,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
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

        tags = ["train_loss", "train_F1", "val_loss", "val_F1", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_logs["Loss"], epoch)
        tb_writer.add_scalar(tags[1], train_logs["f1_score"], epoch)
        tb_writer.add_scalar(tags[2], valid_logs["Loss"], epoch)
        tb_writer.add_scalar(tags[3], valid_logs["f1_score"], epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, log_path)
            print('Model saved!')

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    # use
    # pip install segmentation-models-pytorch
    # pip install pytorch_toolbelt
    # to create env
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="seg", help='seg or cls')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5E-4)
    parser.add_argument('--optimizer', type=str, default='Adam', help='choose from SGD and Adam')
    parser.add_argument('--scheduler', type=str, default='', help='write your lr schedule keywords')

    parser.add_argument('--data-path', type=str, default="Heart Data")
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder backbone')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)