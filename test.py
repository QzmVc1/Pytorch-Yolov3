import torch
import numpy as np


# grid = np.arange(13)
# x, y = np.meshgrid(grid, grid)
# x_offset = torch.FloatTensor(x).view(-1,1)
# y_offset = torch.FloatTensor(y).view(-1,1)
# print(torch.cat((x_offset, y_offset),1).repeat(1, 3).view(-1, 2).unsqueeze(0).shape)

anchors = [(1,2),(3,4),(5,6)]
print(torch.FloatTensor(anchors).repeat(269, 1))
