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
Plain CNN model with two architectural variations (Dense and Sparse) implemented
in PyTorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch.nn as nn
import torch

from nupic.torch.modules import (
    KWinners2d, KWinners, SparseWeights2d, SparseWeights
)


class SimpleCNN(nn.Module):
    """
    The SimpleCNN class creates a simple two layers convolutional neural
    networks that can be parametrized in several way. In particular it offers
    the possibility of adding sparse activations with Kwinners and sparse
    weights with SparseWeights, two layers implemented in nupic.torch.
    """
    def __init__(self, num_classes=10, sparsify=False, percent_on_fc=0.3,
                 percent_on_conv=0.3, k_inference_factor=1.5,
                 boost_strength=1.0, boost_strength_factor=0.9,
                 duty_cycle_period=1000, hidden_units=512, dropout=0.5,
                 weight_sparsity_fc=0.5, weight_sparsity_conv=0.99,
                 image_size=32, channels=3, stats=False):
        """
        Constructor for the object SimpleCNN
            Args:
                num_classes (int): total number of classes of the benchmark,
                                   i.e. maximum output neurons of the model.
                sparsify (bool): if we want to introduce the Kwinners and
                                 SparseWeights layers in the model.
                percent_on_fc (float): Percentage of active units in fc layers.
                percent_on_conv (float): Percentage of active units in convs
                                         layers.
                k_inference_factor (float): boosting parameter. Check the
                                            official Kwinners docs for further
                                            details.
                boost_strength (float): boosting parameter.
                boost_strength_factor (float): boosting parameter.
                hidden_units (int): number of units for the hidden layer.
                dropout (float): dropout probability for each dropout layer.
                weight_sparsity_fc (float): percentage of active weights for
                                            each fc layer.
                weight_sparsity_conv (float): percentage of active weights for
                                              the convs layers.
                image_size (int): input image size.
                channels (int): number of channels in the input.
                stats (bool): if we want to record sparsity statistics.
        """

        super(SimpleCNN, self).__init__()

        self.active_perc_list = []
        self.on_idxs = [0] * hidden_units
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.stats = stats

        if not sparsify:
            self.features = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Flatten(),
                nn.Linear(128 * 24 * 24, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_units, num_classes)
            )

        else:
            self.features = nn.Sequential(
                SparseWeights2d(
                    nn.Conv2d(
                        channels, 64, kernel_size=5, stride=1, padding=0
                    ),
                    weight_sparsity=weight_sparsity_conv
                ),
                nn.ReLU(inplace=True),
                KWinners2d(
                    64, percent_on_conv, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period,
                    local=True
                ),
                nn.Dropout(dropout),
                SparseWeights2d(
                    nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
                    weight_sparsity=weight_sparsity_conv
                ),
                nn.ReLU(inplace=True),
                KWinners2d(
                    128, percent_on_conv, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period,
                    local=True
                ),
                nn.Dropout(dropout),
                nn.Flatten(),
                SparseWeights(
                    nn.Linear(128 * 24 * 24, hidden_units),
                    weight_sparsity=weight_sparsity_fc
                ),
                nn.ReLU(inplace=True),
                KWinners(
                    hidden_units, percent_on_fc, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period),
                nn.Dropout(dropout),

            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_units, num_classes)
            )

    def forward(self, x):
        """
        Forward function for the model inference.
            Args:
                x (tensor): the input tensor to the model.
            Returns:
                tensor: activations of the last fc layer.
        """

        if len(x.size()) < 4:
            x = x.unsqueeze(1)
        x = self.features(x)
        # print(x.size())

        if self.stats:
            # computing active units
            self.active_perc_list.append(
                torch.nonzero(x.data).size(0) /
                (x.size(0) * x.size(1)) * 100
            )

        x = self.classifier(x)

        return x

    def reset_classifier(self):
        """
        Reset the last fc layer. It's useful in a multi-head setting where
        at the end of each task we may want to store the previous head and
        reset it to learn about a new task without much interference.
        """
        self.classifier = nn.Linear(self.hidden_units, self.num_classes)


if __name__ == "__main__":
    """
    Simple instantiation test.
    """

    kwargs = {'num_classes': 10}
    print(SimpleCNN(**kwargs))
