import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def predict_transform(x, inp_size, anchors, class_num, CUDA):
    """
    :param x:             (batch_size, anchor_num*bbox_attrs, grid_size, grid_size)
    :param inp_size:      图像长度
    :param anchors:       锚框图
    :param class_num:     分类个数
    :param CUDA:          是否在CUDA上训练
    :return:              (batch_size, grid_size*grid_size*anchor_num, bbox_attrs)
    """
    # 获取对应属性，准备做维度转换
    batch_size = x.shape[0]
    stride = inp_size // x.shape[2]
    grid_size = inp_size // stride
    anchors_num = len(anchors)
    bbox_attrs = 5 + class_num
    # 维度转换
    x = x.view(batch_size, bbox_attrs*anchors_num, grid_size*grid_size)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, grid_size*grid_size*anchors_num, bbox_attrs)
    # 转化成 grid x grid 上的数值
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    # 确报质点在1x1小方格内
    x[:, :, 0] = torch.sigmoid(x[:, :, 0])
    x[:, :, 1] = torch.sigmoid(x[:, :, 1])
    # 将置信度转化在0-1内
    x[:, :, 4] = torch.sigmoid(x[:, :, 4])
    # 确定质点在 gridxgrid 中的位置，即一二列
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, anchors_num).view(-1, 2).unsqueeze(0)
    # 确定质点位置
    x[:, :, :2] += x_y_offset
    # 确定锚框大小和位置，即三四列
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    # 公式bw=pw×e^tw及bh=ph×e^th，pw为anchorbox的长度
    x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors
    # 计算每个类别的概率（0-1），数值计算在全连接层已实现
    x[:, :, 5:5 + class_num] = torch.sigmoid((x[:, :, 5:5 + class_num]))
    # 将 gridxgrid 的框图恢复到 inp_sizexinp_size 大小
    x[:, :, :4] *= stride
    return x

