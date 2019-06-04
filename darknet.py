import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import predict_transform

"""
1. convolutional
2. shortcut
3. upsample
4. route
5. yolo
6. net
"""
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parseCfg(cfgFile):
    """
    :param cfgFile: 打开cfg文件
    :return: 将每一层的信息变成dict类型并加入列表中，返回包含所有层信息的集合列表
             [{'type':'', 'batch':'xxx',},{'type':'',...},...,{}]
    """
    with open(cfgFile, 'r') as fp:
        lines = fp.read().split('\n')
        lines = [x for x in lines if len(x) > 0 and x[0] != '#']

    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) > 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks

def createModules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    pre_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])                      # param 1
            padding = int(x['pad'])
            kernel_size = int(x['size'])                      # param 2
            stride = int(x['stride'])                          # param 3
            pad = (kernel_size -1) // 2 if padding else 0    # param 4

            # add the convolutional layer
            conv = nn.Conv2d(pre_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"Conv_{index}", conv)

            # add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"BN_{index}", bn)

            # check the activation
            if activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"activation_{index}", act)

        elif x['type'] == "unsample":
            stride = int(x["stride"])
            Unsample = nn.Upsample(scale_factor=stride, mode='nearest')
            module.add_module(f"unsample_{index}",Unsample)

        elif x['type'] == "route":
            digits = x['layers'].split(',')
            start = int(digits[0].strip())
            try:
                end = int(digits[1].strip())
            except:
                end = 0
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            if end < 0:
                filters = output_filters[start + index] + output_filters[end + index]
            else:
                filters = output_filters[start + index]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}',shortcut)

        elif x['type'] == 'yolo':
            masks = x['mask'].split(',')
            masks = [int(i.strip()) for i in masks]
            anchors = x['anchors'].split(',')
            anchors = [int(i.strip()) for i in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in masks]

            detection = DetectionLayer(anchors)
            module.add_module(f'detection_{index}', detection)

        pre_filters = filters
        output_filters.append(filters)
        module_list.append(module)
    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.blocks = parseCfg(cfg_path)
        self.netinfo, self.module_lists = createModules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            if module['type'] == 'convolutional' or module['type'] == 'unsample':
                x = self.module_lists[i](x)

            elif module['type'] == 'route':
                layers = module['layers']
                layers = [int(x.strip()) for x in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 1:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module['type'] == 'shortcut':
                from_ = module['from']
                x = outputs[i-1] + outputs[from_]

            elif module['type'] == 'yolo':
                x = x.data
                anchors = self.module_lists[i][0].anchors
                inp_size = int(self.netinfo['height'])
                class_num = int(module['classes'])
                x = predict_transform(x, inp_size, anchors, class_num, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections

if __name__ == '__main__':
    path = 'C:\SecurityMonitor\Pytorch_Yolov3\cfg\Yolov3.cfg'
    blocks = parseCfg(path)
    net_info, modules = createModules(blocks)
    print(modules[82][0].anchors)