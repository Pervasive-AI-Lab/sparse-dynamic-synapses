#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 8-11-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
Simple Multi-Layer-Perceptron with KWinners in PyTorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn
import torch

from nupic.torch.modules import (
    KWinners,
    SparseWeights
)


class SimpleMLP(nn.Module):

    def __init__(self, sparsify=False, percent_on=0.3,
                 k_inference_factor=1.5, boost_strength=1.0,
                 boost_strength_factor=0.9, duty_cycle_period=1000,
                 num_classes=10, hidden_units=2048, hidden_layers=1,
                 dropout=0.5):
        super(SimpleMLP, self).__init__()

        self.active_perc_list = []
        ft_modules = []

        if sparsify:
            for i in range(hidden_layers):
                if i == 0:
                    ft_modules.append(nn.Linear(28 * 28, hidden_units))
                else:
                    ft_modules.append(nn.Linear(hidden_units, hidden_units))
                ft_modules.append(KWinners(
                    hidden_units, percent_on, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period))
                ft_modules.append(nn.Dropout(dropout))

        else:
            for i in range(hidden_layers):
                if i == 0:
                    ft_modules.append(nn.Linear(28 * 28, hidden_units))
                else:
                    ft_modules.append(nn.Linear(hidden_units, hidden_units))
                ft_modules.append(nn.ReLU(inplace=True))
                ft_modules.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*ft_modules)
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), 28 * 28)
        x = self.features(x)

        # computing active units
        self.active_perc_list.append(
            torch.nonzero(x.data).size(0) / (x.size(0) * x.size(1)) * 100
        )
        # print(self.active_perc_list[-1])

        x = self.classifier(x)
        return x


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs))
