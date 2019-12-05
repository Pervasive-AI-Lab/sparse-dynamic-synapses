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
Plain MLP model with two architectural variations (Dense and Sparse) implemented
in PyTorch.
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
    """
    The SimpleMLP class creates a (multi-)hidden layers neural
    networks that can be parametrized in several way. In particular it offers
    the possibility of adding sparse activations with Kwinners and sparse
    weights with SparseWeights, two layers implemented in nupic.torch.
    """
    def __init__(self, sparsify=False, percent_on=0.3,
                 k_inference_factor=1.5, boost_strength=1.0,
                 boost_strength_factor=0.9, duty_cycle_period=1000,
                 num_classes=10, hidden_units=2048, hidden_layers=1,
                 dropout=0.5, weight_sparsity=0.5, input_size=28*28,
                 stats=False):
        """
        Constructor for the object SimpleCNN
            Args:
                num_classes (int): total number of classes of the benchmark,
                                   i.e. maximum output neurons of the model.
                sparsify (bool): if we want to introduce the Kwinners and
                                 SparseWeights layers in the model.
                percent_on (float): Percentage of active units in fc layers.
                k_inference_factor (float): boosting parameter. Check the
                                            official Kwinners docs for further
                                            details.
                boost_strength (float): boosting parameter.
                boost_strength_factor (float): boosting parameter.
                hidden_units (int): number of units for the hidden layer.
                hidden_layers (int): number of hidden layers.
                dropout (float): dropout probability for each dropout layer.
                weight_sparsity (float): percentage of active weights for
                                         each fc layer.
                input_size (int): input size (assumed a linearized input).
                stats (bool): if we want to record sparsity statistics.
        """

        super(SimpleMLP, self).__init__()

        self.active_perc_list = []
        self.on_idxs = [0] * hidden_units
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.stats = stats

        ft_modules = []

        if sparsify:
            for i in range(hidden_layers):
                if i == 0:
                    ft_modules.append(
                        SparseWeights(
                            nn.Linear(input_size, hidden_units),
                            weight_sparsity=weight_sparsity
                        )
                    )
                else:
                    ft_modules.append(
                        SparseWeights(
                            nn.Linear(hidden_units, hidden_units),
                            weight_sparsity=weight_sparsity
                        )
                    )
                ft_modules.append(KWinners(
                    hidden_units, percent_on, k_inference_factor,
                    boost_strength, boost_strength_factor, duty_cycle_period))
                ft_modules.append(nn.Dropout(dropout))

        else:
            for i in range(hidden_layers):
                if i == 0:
                    ft_modules.append(
                        nn.Linear(input_size, hidden_units)
                    )
                else:
                    ft_modules.append(nn.Linear(hidden_units, hidden_units))
                ft_modules.append(nn.ReLU(inplace=True))
                ft_modules.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*ft_modules)
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        """
        Forward function for the model inference.
            Args:
                x (tensor): the input tensor to the model.
            Returns:
                tensor: activations of the last fc layer.
        """
        x = x.contiguous()
        # print(x.size())
        x = torch.flatten(x, start_dim=1)
        x = self.features(x)

        if self.stats:
            # computing active units
            nonzero = torch.nonzero(x.data)
            self.active_perc_list.append(
                nonzero.size(0) / (x.size(0) * x.size(1)) * 100
            )
            # for mb, unit in nonzero.cpu().numpy():
            #     self.on_idxs[unit] += 1
            # print(self.active_perc_list[-1])

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

    kwargs = {'num_classes': 10}
    print(SimpleMLP(**kwargs))
