#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 7-12-2017                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""

This is the definition od the Mid-caffenet high resolution in Pythorch

"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn

from nupic.torch.modules import (
    KWinners,
    SparseWeights
)


class SimpleMLP(nn.Module):

    def __init__(self, sparsify=False, percent_on = 0.3,
                 k_inference_factor=1.5, boost_strength=1.0,
                 boost_strength_factor=0.9, duty_cycle_period=1000,
                 num_classes=10, hidden_units=2048):
        super(SimpleMLP, self).__init__()

        if sparsify:
            self.features = nn.Sequential(
                nn.Linear(28 * 28, hidden_units),
                KWinners(
                    hidden_units, percent_on, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period),
                nn.Dropout()
            )
            self.classifier = nn.Linear(hidden_units, num_classes)

        else:
            self.features = nn.Sequential(
                nn.Linear(28 * 28, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), 28 * 28)
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs))
