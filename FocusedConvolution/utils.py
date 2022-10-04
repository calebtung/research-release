import copy
import torch.nn as nn
import torch
from focusedconv import FocusedConv2d

def focusify_all_conv2d(m: nn.Module, aoi_mask: torch.Tensor):
    for child_name in m._modules:
        child_m = m._modules[child_name]
        if type(child_m) == nn.Conv2d:
            in_channels = child_m.in_channels
            out_channels = child_m.out_channels
            kernel_size = child_m.kernel_size
            stride = child_m.stride
            padding = child_m.padding
            dilation = child_m.dilation
            groups = child_m.groups
            new_conv = FocusedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, aoi_mask_holder=aoi_mask)
            new_conv.weight = copy.deepcopy(child_m.weight)
            new_conv.bias = copy.deepcopy(child_m.bias)

            m._modules[child_name] = new_conv
        else:
            focusify_all_conv2d(child_m, aoi_mask)