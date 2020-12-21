"""
code from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)


    def forward(self, input, prev_state):

        # get batch and spatial sizes
        batch_size = input.size()[0]
        spatial_size = input.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_size : integer . depth dimensions of hidden state.
        kernel_size : integer. sizes of Conv2d gate kernels.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size
        input_dim = self.input_size

        cell = ConvGRUCell(input_dim, hidden_size, kernel_size)

        self.cells = cell


    def forward(self, x):
        '''
        Parameters
        ----------
        x : 5D input tensor. (batch, time, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (batch, time, channels, height, width).
        '''

        hidden = None

        upd_hidden = []

        N, T, C, H, W = x.size()

        for tidx in range(T):
            hidden = self.cell(x[:, tidx, :, :, :], hidden)
            upd_hidden.append(hidden)

        upd_hidden = torch.stack(upd_hidden, dim=1)

        return upd_hidden