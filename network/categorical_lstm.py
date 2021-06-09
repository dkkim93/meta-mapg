import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from misc.torch_utils import weight_init


def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    """LSTM version of torch.nn.functional
    Since torch.nn.function.lstm does not exist, make same
    functionality to support setattr
    Args:
        input (tensor): Input tensor to LSTM
        hidden (tuple): Tuple that consist of hidden and cell states
        w_ih (tensor): Weight for w_ih
        w_hh (tensor): Weight for w_hh
        b_ih (tensor): Bias for b_ih
        b_hh (tensor): Bias for b_hh
    Reference:
        https://github.com/pytorch/pytorch/blob/\
        6792dac90d468349d58244e0a9a85a5db49a9f40/benchmarks/fastrnns/cells.py
    """
    hx, cx = hidden

    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


class ActorNetwork(nn.Module):
    """Actor network with LSTM that outputs action
    Args:
        input_dim (int): Input dimension to network
        output_dim (int): Output dimension of network
        name (str): Prefix for each layer
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, input_dim, output_dim, name, args):
        super(ActorNetwork, self).__init__()

        setattr(self, name + "_actor_l1", nn.Linear(input_dim, args.n_hidden))
        setattr(self, name + "_actor_l2", nn.LSTMCell(args.n_hidden, args.n_hidden))
        setattr(self, name + "_actor_l3", nn.Linear(args.n_hidden, output_dim))
        self.name = name + "_actor"
        self.apply(weight_init)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x, (hx, cx) = x

        x = F.linear(x, weight=params[self.name + "_l1.weight"], bias=params[self.name + "_l1.bias"])
        x = F.relu(x)
        hy, cy = lstm_cell(
            input=x,
            hidden=(hx, cx),
            w_ih=params[self.name + "_l2.weight_ih"],
            w_hh=params[self.name + "_l2.weight_hh"],
            b_ih=params[self.name + "_l2.bias_ih"],
            b_hh=params[self.name + "_l2.bias_hh"])
        x = F.linear(hy, weight=params[self.name + "_l3.weight"], bias=params[self.name + "_l3.bias"])
        x = F.softmax(x, dim=1)

        return x, (hy, cy)


class ValueNetwork(nn.Module):
    """Value network with LSTM that outputs value (V)
    Args:
        input_dim (int): Input dimension to network
        name (str): Prefix for each layer
        args (argparse): Python argparse that contains arguments
    """
    def __init__(self, input_dim, name, args):
        super(ValueNetwork, self).__init__()

        setattr(self, name + "_value_l1", nn.Linear(input_dim, args.n_hidden))
        setattr(self, name + "_value_l2", nn.LSTMCell(args.n_hidden, args.n_hidden))
        setattr(self, name + "_value_l3", nn.Linear(args.n_hidden, 1))
        self.name = name + "_value"
        self.apply(weight_init)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x, (hx, cx) = x

        x = F.linear(x, weight=params[self.name + "_l1.weight"], bias=params[self.name + "_l1.bias"])
        x = F.relu(x)
        hy, cy = lstm_cell(
            input=x,
            hidden=(hx, cx),
            w_ih=params[self.name + "_l2.weight_ih"],
            w_hh=params[self.name + "_l2.weight_hh"],
            b_ih=params[self.name + "_l2.bias_ih"],
            b_hh=params[self.name + "_l2.bias_hh"])
        x = F.linear(hy, weight=params[self.name + "_l3.weight"], bias=params[self.name + "_l3.bias"])

        return x, (hy, cy)
