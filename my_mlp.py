from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, layer_number, l2_strength=1e-7,
                 with_dropout=False, with_batch_norm=True, with_residual=True):
        super(MLPClassifier, self).__init__()

        self.l2_strength = l2_strength
        self.with_dropout = with_dropout
        self.with_batch_norm = with_batch_norm
        self.with_residual = with_residual

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h_weights = [nn.Linear(hidden_size,hidden_size) for l in range(layer_number-1)]
        for index,hw in enumerate(self.h_weights):
            self.add_module('h_weights[%d]'%index,hw)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        if self.with_batch_norm:
            self.norms = [torch.nn.BatchNorm1d(hidden_size) for l in range(layer_number)]
            for index,bn in enumerate(self.norms):
                self.add_module('batch_norms[%d]'%index,bn)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        if self.with_batch_norm:
            h1 = self.norms[0](h1)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        h = h1
        for index,hw in enumerate(self.h_weights):
            tmp_h = hw(h)
            if self.with_batch_norm:
                tmp_h = self.norms[index+1](tmp_h)
            tmp_h = F.relu(tmp_h)
            if self.with_dropout:
                tmp_h = F.dropout(tmp_h, training=self.training)
            if self.with_residual:
                h = h + tmp_h
            else:
                h = tmp_h

        logits = self.h2_weights(h)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            l2_loss = torch.sum(torch.tensor([torch.sum(hw.weight*hw.weight)
                                 for hw in [self.h1_weights,self.h2_weights]+self.h_weights]))
            loss = F.nll_loss(logits, y) + l2_loss

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits
