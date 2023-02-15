import os
import math
import argparse
import datetime
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import autoaugment
from torch.utils.data.sampler import WeightedRandomSampler

from model import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_images_path, train_images_label, val_images_path, val_images_label, cls_num = read_split_data(args.data_path, test_datasets="datasets_test")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomChoice([
                                         transforms.RandomHorizontalFlip(p=0.2),
                                         transforms.RandomVerticalFlip(p=0.2),
                                         transforms.RandomRotation(degrees=45)]),
                                     transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                     autoaugment.TrivialAugmentWide(),
                                     transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.CenterCrop(img_size[num_model][0]), #augmentation when classifying
                                     transforms.Resize(img_size[num_model][0]),
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

    weights = torch.tensor(cls_num, dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    class_weights = weights / weights.sum()
    weight_list = [class_weights[i] for i in train_images_label]
    weighted_sampler = WeightedRandomSampler(weights=weight_list, num_samples=len(train_images_path))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               # sampler=weighted_sampler,
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
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "" and args.resume == "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("No weights file: {}".format(args.weights))
    elif args.weights != "" and args.resume != "":
        raise ValueError("Set weights as '' if you wanna start from checkpoint")

    model01 = create_model(num_classes=2).to(device)
    model12 = create_model(num_classes=2).to(device)
    model23 = create_model(num_classes=2).to(device)
    model_bi = [model01, model12, model23]
    weights_list = ["weights/best_weight_01.pth", "weights/best_weight_12.pth", "weights/best_weight_23.pth"]
    for i, path in enumerate(weights_list):
        if os.path.exists(path):
            weights_dict = torch.load(path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model_bi[i].state_dict()[k].numel() == v.numel()}
            print(model_bi[i].load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("No weights file: {}".format(model_bi[i]))

    # 冻结权重
    layers_to_train = "39"
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if layers_to_train not in name:
                para.requires_grad_(False)
            else:
                break
    pg = [p for p in model.parameters() if p.requires_grad]
    print([l for l, p in model.named_parameters() if p.requires_grad is True])

    # 优化器调度器
    if args.optimizer == 'SGD':
        # SGD
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    else:
        # Adam
        optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-6)

    def lr_lambda(current_step: int):
        num_warmup_steps = 2 if "warmup" in args.scheduler else -1
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if "cosine" in args.scheduler:
            return ((1 + math.cos(current_step * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        elif "steps" in args.scheduler:
            lrf_step = [1, 0.2, 0.02, 0.002, 0.0005]
            return lrf_step[int(current_step / 6)]
        return max(1E-4, pow(0.6, int(current_step/4)))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)  # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    # 训练参数和记录存储
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if os.path.exists("./runs") is False:
        os.makedirs("./runs")
    log_path = "./runs/{}".format(datetime.datetime.now().strftime("%Y_%m%d-%H_%M_%S"))

    start_epoch = 0
    best_val_acc = 0

    if args.resume != "":
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            log_path = checkpoint['log_path']
            best_val_acc = checkpoint['best_val_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            if 'reset' not in args.scheduler:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'reset' not in args.scheduler:
                scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            assert True, "Fail to load checkpoint, check its path."

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_writer = SummaryWriter(log_dir=log_path)
    with open(log_path + "/arg_list_epoch[{}].json".format(start_epoch), "w") as f:
        f.write(json.dumps(vars(args)))
    results_file = os.path.join(log_path, "err_list.txt")

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                loss_f=torch.as_tensor(class_weights, dtype=torch.float32).to(device),
                                                info_path=results_file)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     loss_f=torch.as_tensor(class_weights, dtype=torch.float32).to(device),
                                     model_bi=[model01, model12, model23],
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
        torch.save(checkpoint, log_path + f"/checkpoint_from_epoch[{start_epoch}].pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), log_path + "/best_weight.pth")

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
