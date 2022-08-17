import os
import sys
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
import pandas as pd
from torchvision import transforms
from my_dataset import MyDataSet

def read_split_data(root: str, test_rate: float = 0.2 , n_class = 2 , seed = 2):
    random.seed(seed)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    if n_class == 3:
        pa_class = ["covid","normal","others"]
    elif n_class == 2:
        pa_class = ["covid","normal"]
    class_indices = dict((k, v) for v, k in enumerate(pa_class))

    train_images_path = []  
    train_images_label = []  
    test_images_path = []  
    test_images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"] 

    for cla in pa_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        test_path = random.sample(images, k=int(len(images) * test_rate))

        for img_path in images:
            if img_path in test_path: 
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for testing.".format(len(test_images_path)))
    assert len(train_images_path) > 0, "not find data for train."
    assert len(test_images_path) > 0, "not find data for etest"

    return train_images_path, train_images_label, test_images_path, test_images_label

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels = labels.long()
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

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # count tp+tn
    accu_loss = torch.zeros(1).to(device)  # count losses

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels = labels.long()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def cal_m(model, data_loader, device, alo='def', num=2):
    model.eval()
    sample_num = 0
    label = np.array([])
    score = np.array([])
    pre_label = np.array([])
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        pre_label_t = pred_classes.cpu().numpy()
        if num == 2:
            score_t = torch.softmax(pred, 1).cpu().numpy()
            score_t = score_t[:, 1]
            label = np.append(label, labels.numpy())
            score = np.append(score, score_t)
            pre_label = np.append(pre_label, pre_label_t)
        else:
            score_t = torch.softmax(pred, 1).cpu().numpy()
            label = np.append(label, labels.numpy())
            score = np.append(score, score_t)
            pre_label = np.append(pre_label, pre_label_t)

    if num == 2:
        fpr, tpr, threshold = roc_curve(label, score)
        pre, rec_, _ = precision_recall_curve(label, score)
        acc = accuracy_score(label, pre_label)
        rec = recall_score(label, pre_label)
        f1 = f1_score(label, pre_label)
        Pre = precision_score(label, pre_label)
        au = auc(fpr, tpr)
        apr = auc(rec_, pre)
        # label ture_label，score predict_score，pre_label predict_label from predict_score
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write(str(round(Pre, 4)) + '\t')
        f.write(str(round(rec, 4)) + '\t')
        f.write(str(round(acc, 4)) + '\t')
        f.write(str(round(f1, 4)) + '\t')
        f.write(str(round(au, 4)) + '\t')
        f.write(str(round(apr, 4)) + '\n\n')
        f.close()
        print('Precision is :{}'.format(Pre))
        print('Recall is :{}'.format(rec))
        print("ACC is: {}".format(acc))
        print("F1 is: {}".format(f1))
        print("AUC is: {}".format(au))
        print('AUPR is :{}'.format(apr))
    else:
        acc = accuracy_score(label, pre_label)
        score = score.reshape((-1, 3))
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write("ACC: " + str(round(acc, 4)) + '\n')
        f.close()
        print("ACC is: {}".format(acc))
    return score

def evaluate(score,label,pre_label,num,alo):
    if num == 2:
        fpr, tpr, threshold = roc_curve(label, score)
        pre, rec_, _ = precision_recall_curve(label, score)
        acc = accuracy_score(label, pre_label)
        rec = recall_score(label, pre_label)
        f1 = f1_score(label, pre_label)
        Pre = precision_score(label, pre_label)
        au = auc(fpr, tpr)
        apr = auc(rec_, pre)
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write(str(round(Pre, 4)) + '\t')
        f.write(str(round(rec, 4)) + '\t')
        f.write(str(round(acc, 4)) + '\t')
        f.write(str(round(f1, 4)) + '\t')
        f.write(str(round(au, 4)) + '\t')
        f.write(str(round(apr, 4)) + '\n\n')
        f.close()
        print('Precision is :{}'.format(Pre))
        print('Recall is :{}'.format(rec))
        print("ACC is: {}".format(acc))
        print("F1 is: {}".format(f1))
        print("AUC is: {}".format(au))
        print('AUPR is :{}'.format(apr))
    else:
        acc = accuracy_score(label, pre_label)
        score = score.reshape((-1, 3))
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write("ACC: " + str(round(acc, 4)) + '\n')
        f.close()
        print("ACC is: {}".format(acc))
    return score

def mk_dir(j):
    if os.path.exists('./res_dir/label/dataset' + str(j)) is False:
        os.makedirs('./res_dir/label/dataset' + str(j))
    if os.path.exists('./res_dir/train_res/dataset' + str(j)) is False:
        os.makedirs('./res_dir/train_res/dataset' + str(j))
    if os.path.exists('./res_dir/weights/dataset' + str(j)) is False:
        os.makedirs('./res_dir/weights/dataset' + str(j))
    if os.path.exists('./res_dir/best_res') is False:
        os.makedirs('./res_dir/best_res')
    if os.path.exists('./init') is False:
        os.makedirs('./init')
    
