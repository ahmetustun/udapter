import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class LinearWithPGN(torch.nn.Module):

    def __init__(self, in_params, in_features, out_features, bias=True):
        super(LinearWithPGN, self).__init__()
        self.in_params = in_params
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_params, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_params, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, pgn_vector):
        project_w = torch.matmul(pgn_vector, self.weight.view(self.in_params, -1)).view(self.out_features, self.in_features)
        project_b = None
        if self.bias is not None:
            project_b = torch.matmul(pgn_vector, self.bias)

        return F.linear(input, project_w, project_b)

    def extra_repr(self):
        return 'in_params={}, in_features={}, out_features={}, bias={}'.format(
            self.in_params, self.in_features, self.out_features, self.bias is not None
        )


class BilinearWithPGN(nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features', 'bias']

    def __init__(self, in_params, in1_features, in2_features, out_features, bias=True):
        super(BilinearWithPGN, self).__init__()
        self.in_params = in_params
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_params, out_features, in1_features, in2_features))

        if bias:
            self.bias = Parameter(torch.Tensor(in_params, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2, pgn_vector):
        weight = torch.matmul(pgn_vector, self.weight.view(self.in_params, -1)).view(self.out_features, self.in1_features, self.in2_features)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(pgn_vector, self.bias)
        return F.bilinear(input1, input2, weight, bias)

    def extra_repr(self):
        return 'in_params={}, in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in_params, self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )
