
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np


def to_var(tensor, volatile=False):
    tensor = Variable(tensor, volatile=volatile)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        # size = m.weight.size()
        # fan_out = size[0]  # number of rows
        # fan_in = size[1]  # number of columns
        # variance = np.sqrt(2.0 / (fan_in + fan_out))
        # m.weight.data.normal_(0.0, variance)

        m.weight.data.uniform_(-0.1, 0.1)

        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.fill_(0)


def mask_tensor(tensor, mask, mask_value=0.0):
    tensor[mask[:, :, None].expand_as(tensor)] = mask_value
    return tensor


def select_indices(tensor, indices, dim):
    len_size = len(tensor.size())
    indices = indices[:, None, None].repeat(1, 1, tensor.size(-1))
    output = tensor.gather(dim, indices)
    return output
