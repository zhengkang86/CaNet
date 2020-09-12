from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os.path as osp
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
import argparse
from dataset_mask_train import Dataset as Dataset_train
from dataset_mask_val import Dataset as Dataset_val
import os
import torch
from one_shot_network import Res_Deeplab
import torch.nn as nn
import numpy as np

from fss import FewShotInstData
import sys
sys.path.append(".")
from coco_for_canet import COCO_CaNet

import torchvision.transforms.functional as transforms_F
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00025)
parser.add_argument('-prob',
                    type=float,
                    help='dropout rate of history mask',
                    default=0.7)
parser.add_argument('-workers',
                    type=int,
                    help='workers',
                    default=12)
parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=48)
parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=48)
parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=0)
parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0')
parser.add_argument('-iter_time',
                    type=int,
                    default=5)
options = parser.parse_args()

coco_dir = '/home/kang/Projects/data/COCO'

# set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

torch.backends.cudnn.benchmark = True

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 2
learning_rate = options.lr  # 0.000025#0.00025
input_size = (321, 321)
batch_size = options.bs
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True

# Create network.
model = Res_Deeplab(num_classes=num_class)
# load resnet-50 pretrained parameter
model = load_resnet50_param(model, stop_layer='layer4')
model = nn.DataParallel(model)

# disable the  gradients of not optomized layers
turn_off(model)

checkpoint_dir = 'checkpoint/fold_%d/' % options.fold
check_dir(checkpoint_dir)

# loading data
folds = [0, 1, 2]
val_folds = [3]

# trainset
exclude_files = './json/bad_inst_list.json'
# dataset = COCO_CaNet(coco_dir, 'train2017', folds, 1, exclude_list_file=exclude_files, normalize_mean=IMG_MEAN, normalize_std=IMG_STD)
# # dataset = Dataset_train(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
# #                         normalize_std=IMG_STD, prob=options.prob)
# trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=options.workers)

# valset
# this only a quick val dataset where all images are 321*321.
valset = COCO_CaNet(coco_dir, 'val2017', val_folds, 1, normalize_mean=IMG_MEAN, normalize_std=IMG_STD)
# valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
#                      normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=options.workers,
                            drop_last=False)

optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * learning_rate}],
                      lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

loss_list = []  # track training loss
iou_list = []  # track validaiton iou
highest_iou = 0

model.cuda()
# load model weights
state_dict = torch.load('/home/kang/Projects/FSOS/CaNet/checkpoint/fold_0/model/epoch_1.pth')
model.load_state_dict(state_dict)

n_val_class = 20  # number of classes for evaluation, which is 5 for VOC, 20 for COCO

# inference and evaluation
# ======================evaluate now==================
with torch.no_grad():
    print('----Evaluation----')
    model = model.eval()

    # valset.history_mask_list = [None] * 1000
    best_iou = 0
    for eva_iter in range(options.iter_time):
        all_inter, all_union, all_predict = [0] * n_val_class, [0] * n_val_class, [0] * n_val_class
        for i_iter, batch in enumerate(tqdm.tqdm(valloader)):

            query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch

            query_rgb = query_rgb.cuda(0)
            support_rgb = support_rgb.cuda(0)
            support_mask = support_mask.cuda(0)
            query_mask = query_mask.cuda(0).long()  # change formation for crossentropy use
            query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
            history_mask = history_mask.cuda(0)

            pred = model(query_rgb, support_rgb, support_mask, history_mask)
            pred_softmax = F.softmax(pred, dim=1).data.cpu()

            """==========================="""
            # visualization
            # IMG_MEAN_UNNORM = -(torch.tensor(IMG_MEAN) / torch.tensor(IMG_STD)).to(support_rgb.device)
            # IMG_STD_UNNORM = (torch.tensor([1, 1, 1]) / torch.tensor(IMG_STD)).to(support_rgb.device)
            # pred_softmax = nn.functional.interpolate(pred_softmax, size=input_size, mode='bilinear', align_corners=True)  # upsample
            # _, pred_label = torch.max(pred_softmax, 1)

            # for j in range(support_mask.shape[0]):
            #     s_rgb = transforms_F.normalize(support_rgb[j, ...], IMG_MEAN_UNNORM, IMG_STD_UNNORM).clamp(0, 1)
            #     q_rgb = transforms_F.normalize(query_rgb[j, ...], IMG_MEAN_UNNORM, IMG_STD_UNNORM).clamp(0, 1)
            #     fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            #     ax[0, 0].imshow(s_rgb.permute((1, 2, 0)).detach().cpu().numpy())
            #     ax[0, 1].imshow(support_mask[j, 0, :, :].detach().cpu().numpy(), cmap='gray')
            #     ax[0, 2].imshow(support_mask[j, 0, :, :].detach().cpu().numpy(), cmap='gray')
            #     ax[1, 0].imshow(q_rgb.permute((1, 2, 0)).detach().cpu().numpy())
            #     ax[1, 1].imshow(query_mask[j, :, :].detach().cpu().numpy(), cmap='gray')
            #     ax[1, 2].imshow(pred_label[j, :, :].detach().cpu().numpy(), cmap='gray')
            #     plt.show()
            """==========================="""

            # update history mask
            for j in range(support_mask.shape[0]):
                sub_index = index[j]
                valset.history_mask_list[sub_index] = pred_softmax[j]
                pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                    align_corners=True)  # upsample  # upsample

            _, pred_label = torch.max(pred, 1)
            inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)

            for j in range(query_mask.shape[0]):  # batch size
                all_inter[sample_class[j] - (options.fold * n_val_class + 1)] += inter_list[j]
                all_union[sample_class[j] - (options.fold * n_val_class + 1)] += union_list[j]

        IOU = [0] * n_val_class
        for j in range(n_val_class):
            IOU[j] = all_inter[j] / all_union[j]

        mean_iou = np.mean(IOU)
        print('IOU:%.4f' % mean_iou)
        # if mean_iou > best_iou:
        #     best_iou = mean_iou
        # else:
        #     break
