import os
import sys
import json
import pickle
import random
import datetime
import numpy
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def read_split_data(root: str, val_rate: float = 0.2, test_datasets: str = ""):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    val_rate = 0 if test_datasets != "" else val_rate

    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in classes:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        if test_datasets == "":
            val_path = random.sample(images, k=int(len(images) * val_rate))
        else:
            test_path = os.path.join(test_datasets, cla)
            # 遍历获取supported支持的所有文件路径
            val_path = [os.path.join(test_datasets, cla, i) for i in os.listdir(test_path)
                      if os.path.splitext(i)[-1] in supported]
            images = images + val_path
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False # 想看样本分布把这改成True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(classes)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(classes)), classes)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label, every_class_num

class Estimate():
    def __init__(self, num_classes:int, is_train=True):
        self.predict_list = []
        self.label_list = []
        self.name_list = []
        self.err_name_list = []
        self.prob_list = []
        self.num_in_classes = torch.zeros(num_classes)
        self.pre_in_classes = torch.zeros(num_classes)
        self.true_labels = torch.zeros(num_classes)
        self.precision = torch.zeros(num_classes)
        self.recall = torch.zeros(num_classes)

    def add(self, pred_classes, labels, names, prob):
        self.predict_list.extend(pred_classes)
        self.label_list.extend(labels)
        self.name_list.extend(names)
        self.prob_list.extend(prob)

    def cal(self):
        for x, y, n, p in zip(self.predict_list, self.label_list, self.name_list, self.prob_list):
            self.pre_in_classes[x] += 1
            self.num_in_classes[y] += 1
            if x != y:
                self.err_name_list.append((n, y, x, p)) # [图片名, 原始标签, 错判标签]
            else:
                self.true_labels[x] += 1

        self.precision = torch.div(self.true_labels, self.pre_in_classes)
        self.recall = torch.div(self.true_labels, self.num_in_classes)

        return self.precision, self.recall, self.err_name_list

    def save(self, epoch, path, is_train=True):
        p = self.precision.cpu().numpy().tolist()
        r = self.recall.cpu().numpy().tolist()
        state = "train" if is_train else "valid"
        with open(path, "a") as f:
            info = f"[epoch({state}): {epoch} ]\n" \
                   f"precision: {p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}, {p[3]:.6f}\n" \
                   f"recall: {r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}\n" \
                   f"err_name_list [图片名, 原始标签, 错判标签]:\n"
            err_list = ""
            for i in self.err_name_list:
                name = i[0].split("\\")
                name = name[-2]+"/"+name[-1]
                err_list += f"path: {name}    {i[1]}    {i[2]}    {i[3]}\n"

            f.write(info + err_list + "\n")


def train_one_epoch(model, optimizer, data_loader, device, epoch, info_path, loss_f=None):
    model.train()
    estimator = Estimate(4)

    loss_function = torch.nn.CrossEntropyLoss(weight=loss_f)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, names = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        estimator.add(pred_classes.cpu().numpy().tolist(), labels.cpu().numpy().tolist(), names, pred.cpu().detach().numpy().tolist())

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    estimator.cal()
    estimator.save(epoch, info_path)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, info_path, loss_f=None, model_bi=[]):
    model.eval()
    for m in model_bi:
        m.eval()
    estimator = Estimate(4)

    loss_function = torch.nn.CrossEntropyLoss(weight=loss_f)
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, names = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        prob = torch.softmax(pred, dim=1)
        for i, p in enumerate(prob):
            prob[i] = p/prob[i].sum()

        prob_ = prob.clone()
        for i, p in enumerate(pred_classes):
            first_max = prob[i][p]
            prob_[i][p] = 0
            second_max, second_label = torch.max(prob_[i], dim=0)
            if first_max-second_max < 0.25:
                if (p == 0 and second_label == 1):
                    p_ = model_bi[0](images[i].unsqueeze(0).to(device))
                    c = torch.max(p_, dim=1)[1]
                    pred_classes[i] = c
                elif (p == 1 and second_label == 2) or (p == 2 and second_label == 1):
                    p_ = model_bi[1](images[i].unsqueeze(0).to(device))
                    c = torch.max(p_, dim=1)[1] + 1
                    pred_classes[i] = c
                elif (p == 3 and second_label == 2):
                    p_ = model_bi[2](images[i].unsqueeze(0).to(device))
                    c = torch.max(p_, dim=1)[1] + 2
                    pred_classes[i] = c

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        estimator.add(pred_classes.cpu().numpy().tolist(), labels.cpu().numpy().tolist(), names, prob.cpu().detach().numpy().tolist())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    estimator.cal()
    estimator.save(epoch, info_path, is_train=False)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
