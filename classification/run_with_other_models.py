import os
import math
import argparse
import datetime
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import efficientnetv2_m as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./train_log") is False:
        os.makedirs("./train_log")

    train_images_path, train_images_label, val_images_path, val_images_label, cls_num = read_split_data(args.data_path)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    # 预训练权重
    model = models.resnext101_32x8d().to(device)
    if args.weights != "" and args.resume == "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("No weights file: {}".format(args.weights))

    layer_list = []
    count = 0
    # for k in model.children():
    #     count += 1
    #     layer_list.append([count, k])
    # with open("resnext101_layer_list.txt", "w") as f:
    #     f.write(str(layer_list))

    for k in model.children():
        count += 1
        if count > 7:
            ...
        else:
            for param in k.parameters():
                param.requires_grad = False
    input_features = model.fc.in_features
    model.fc = torch.nn.Linear(input_features, 4).to(device)

    # 优化器调度器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    # 训练参数和记录存储
    log_path = "./train_log/{}".format(datetime.datetime.now().strftime("%Y_%m%d-%H_%M_%S"))
    start_epoch = 0
    best_val_acc = 0

    if args.resume != "":
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            log_path = checkpoint['log_path']
            best_val_acc = checkpoint['best_val_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            assert True, "Fail to load checkpoint, check its path."

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    with open(log_path + "/arg_list.json", "w") as f:
        f.write(json.dumps(vars(args)))
    results_file = os.path.join(log_path, "err_list.txt")

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                cls_num=cls_num,
                                                info_path=results_file)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     info_path=results_file)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        checkpoint = {
            'log_path': log_path,
            'best_val_acc': best_val_acc,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            }
        torch.save(checkpoint, log_path + "/checkpoint.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), log_path + "/best_weight.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="datasets")

    # load model weights
    parser.add_argument('--weights', type=str, default='resnext101_32x8d-8ba56ff5.pth', help='initial weights path')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
