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
Plain CNN model with KWinners implemented in PyTorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn
import torch

from nupic.torch.modules import (
    KWinners2d, KWinners
)

from nupic.research.frameworks.pytorch.modules import (
    KWinners2dLocal
)


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10, sparsify=False, percent_on=0.3,
                 k_inference_factor=1.5, boost_strength=1.0,
                 boost_strength_factor=0.9, duty_cycle_period=1000,
                 hidden_units=512, dropout=0.5):
        super(SimpleCNN, self).__init__()

        self.active_perc_list = []

        if not sparsify:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 24 * 24, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, num_classes)
            )

        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                KWinners2dLocal(
                    64, percent_on, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period),
                nn.Dropout(dropout),

            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 24 * 24, hidden_units),
                nn.ReLU(inplace=True),
                KWinners(
                    hidden_units, percent_on, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, num_classes)
            )

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.features(x)
        # print(x.size())

        # computing active units
        self.active_perc_list.append(
            torch.nonzero(x.data).size(0) /
            (x.size(0) * x.size(1) * x.size(2) * x.size(3)) * 100
        )
        # print(self.active_perc_list[-1])
        x = x.view(-1, 64 * 24 * 24)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleCNN(**kwargs))
