import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from einops import rearrange

from src.utils.utils import create_directory
from src.utils.utils import get_test_loss_acc
from src.utils.utils import save_models
from src.utils.utils import log_history
from src.utils.utils import calculate_metrics
from src.utils.utils import save_logs
from src.utils.utils import model_predict
from src.utils.utils import plot_epochs_metric
import os
import torchvision
import torch.nn.functional as F


class LKRFN(nn.Module):

    def __init__(self, in_channels, out1_channels_1, out1_channels_2, out1_channels_3,
                 out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3,
                 out2_channels_4, num_class):
        super(LKRFN, self).__init__()

        in_channels_module1 = in_channels
        self.stem_1 = Inception_module(in_channels, out1_channels_1, out1_channels_2,
                                       out1_channels_3, out1_channels_4)
        in_channels_module2 = out1_channels_1 + out1_channels_2 + out1_channels_3 + out1_channels_4
        self.module1 = BasicRFB_s(in_channels_module2, in_channels_module2, 15, 3)

        in_channels_module3 = out2_channels_1 + out2_channels_2 + out2_channels_3 + out2_channels_4
        self.module2 = BasicRFB_s(in_channels_module2, in_channels_module2, 15, 3)

        self.module3 = BasicRFB_s(in_channels_module2, in_channels_module2, 15, 3)

        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels_module2, num_class)

    def forward(self, x):
        x = self.stem_1(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_ave_pooling(x).squeeze()

        output = self.linear(x)

        return output, x


class Inception_module(nn.Module):

    def __init__(self, in_channels, out_channels_1, out_channels_2,
                 out_channels_3, out_channels_4, **kwargs):
        super(Inception_module, self).__init__()

        self.conv_1 = BasicConv2d(in_channels, out_channels_1, 1)

        self.conv_3 = BasicConv2d(in_channels, out_channels_2, 3)

        self.conv_3_3_a = BasicConv2d(in_channels, out_channels_3, 3)
        self.conv_3_3_b = BasicConv2d(out_channels_3, out_channels_3, 3)

        self.conv_3pool_1_3 = nn.MaxPool2d(3, 1, 3 // 2)
        self.conv_3pool_1_1 = BasicConv2d(in_channels, out_channels_4, 1)

    def forward(self, x):
        branch_1 = self.conv_1(x)

        branch_2 = self.conv_3(x)

        branch_3 = self.conv_3_3_a(x)
        branch_3 = self.conv_3_3_b(branch_3)

        branch_4 = self.conv_3pool_1_1(x)
        branch_4 = self.conv_3pool_1_3(branch_4)

        module_outputs = [branch_1, branch_2, branch_3, branch_4]

        return torch.cat(module_outputs, 1)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_s(nn.Module):

    def __init__(self, in_planes, out_planes, large_kernel, small_kernel, stride=1, scale=0.1):
        super(BasicRFB_s, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        padding_large = (large_kernel - 1) // 2
        padding_small = (small_kernel - 1) // 2

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch4_0_1 = BasicConv(in_planes, inter_planes, kernel_size=1, stride=1)
        self.branch4_0_3 = BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        self.branch4_1 = conv_bn(in_channels=inter_planes, out_channels=inter_planes,
                                 kernel_size=(large_kernel, small_kernel),
                                 stride=1, padding=(padding_large, padding_small), dilation=1, groups=inter_planes,
                                 bn=True)
        self.branch4_2 = conv_bn(in_channels=inter_planes, out_channels=inter_planes,
                                 kernel_size=(small_kernel, large_kernel),
                                 stride=1, padding=(padding_small, padding_large), dilation=1, groups=inter_planes,
                                 bn=True)
        self.branch4_3 = BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=15, dilation=15, relu=False)

        self.ConvLinear = BasicConv(5 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x4_0_1 = self.branch4_0_1(x)
        x4_0_3 = self.branch4_0_3(x4_0_1)
        x4_1 = self.branch4_1(x4_0_1)
        x4_2 = self.branch4_2(x4_0_1)
        x4_3 = self.relu(x4_1 + x4_2 + x4_0_3)
        x4_3 = self.branch4_3(x4_3)

        out = torch.cat((x0, x1, x2, x3, x4_3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


use_sync_bn = False


def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))

    if bn:
        result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def train_op(classifier_obj, EPOCH, batch_size, LR, train_x, train_y,
             test_x, test_y, output_directory_models,
             model_save_interval, test_split,
             save_best_train_model=True,
             save_best_test_model=True):
    # prepare training_data
    BATCH_SIZE = int(min(train_x.shape[0] / 8, batch_size))
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset=torch_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=drop_last_flag
                                   )

    # init lr&train&test loss&acc log
    lr_results = []
    loss_train_results = []
    accuracy_train_results = []
    loss_test_results = []
    accuracy_test_results = []

    # prepare optimizer&scheduler&loss_function
    optimizer = torch.optim.Adam(classifier_obj.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                           patience=50,
                                                           min_lr=0.000002, verbose=True)
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # save init model
    output_directory_init = output_directory_models + 'init_model.pkl'
    torch.save(classifier_obj.state_dict(), output_directory_init)  # save only the init parameters

    training_duration_logs = []
    start_time = time.time()
    for epoch in range(EPOCH):

        # loss_sum_train = torch.tensor(0)
        # true_sum_train = torch.tensor(0)

        for step, (x, y) in enumerate(train_loader):
            """batch_x = x.cuda()
            batch_y = y.cuda()"""
            batch_x = x.data.cpu()
            batch_y = y.data.cpu()
            output_bc = classifier_obj(batch_x)[0]

            # # cal the num of correct prediction per batch
            # pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            # true_num_bc = torch.sum(pred_bc == batch_y).data

            # cal the sum of pre loss per batch
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test per epoch
        classifier_obj.eval()
        loss_train, accuracy_train = get_test_loss_acc(classifier_obj, loss_function, train_x, train_y, 15)
        loss_test, accuracy_test = get_test_loss_acc(classifier_obj, loss_function, test_x, test_y, test_split)
        classifier_obj.train()

        # update lr
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']

        ######################################dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(classifier_obj.eval(), loss_function, train_x, train_y, test_split)

        # loss_test, accuracy_test = get_loss_acc(classifier_obj.eval(), loss_function, test_x, test_y, test_split)

        # classifier_obj.train()
        ##################################################################################

        # log lr&train&test loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_test_results.append(loss_test)
        accuracy_test_results.append(accuracy_test)

        # print training process
        if (epoch + 1) % 1 == 0:
            print('Epoch:', (epoch + 1), '|lr:', lr,
                  '| train_loss:', loss_train,
                  '| train_acc:', accuracy_train,
                  '| test_loss:', loss_test,
                  '| test_acc:', accuracy_test)

        training_duration_logs = save_models(classifier_obj, output_directory_models,
                                             loss_train, loss_train_results,
                                             accuracy_test, accuracy_test_results,
                                             model_save_interval, epoch, EPOCH,
                                             start_time, training_duration_logs,
                                             save_best_train_model, save_best_test_model)

        # save last_model
    output_directory_last = output_directory_models + 'last_model.pkl'
    torch.save(classifier_obj.state_dict(), output_directory_last)  # save only the init parameters

    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                          loss_test_results, accuracy_test_results)

    return (history, training_duration_logs)

