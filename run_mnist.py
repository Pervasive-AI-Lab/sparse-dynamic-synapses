#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. ContinualAI. All rights reserved.                        #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-10-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
Starting example using avalanche and the naive strategy on Permuted MNIST.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from models.simple_mlp import SimpleMLP
from benchmarks.cmnist import CMNIST
from utils.pytorch_utils import train_net, test_multitask

import torch
import configparser
from pprint import pprint


if __name__ == "__main__":

    import sys
    print(sys.path)

    # setup parameters
    cfg_loc = 'exps/'
    exp_name = 'Exp1'
    config = configparser.ConfigParser()
    config.read(cfg_loc + "exps_params.cfg")

    print(dict(config))
    exp_config = config[exp_name]
    pprint(exp_config)

    mb_size = int(exp_config['mb_size'])
    train_ep = int(exp_config['train_ep'])
    use_cuda = True
    preproc = None
    model = SimpleMLP(
        sparsify=exp_config.getboolean('sparsify'),
        percent_on=float(exp_config['percent_on']),
        k_inference_factor=float(exp_config['k_inference_factor']),
        boost_strength=float(exp_config['boost_strength']),
        boost_strength_factor=float(exp_config['boost_strength_factor']),
        duty_cycle_period=int(exp_config['duty_cycle_period']),
        num_classes=int(exp_config['num_classes']),
        hidden_units=int(exp_config['hidden_units'])
    )
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(exp_config['lr']),
        nesterov=True,
        momentum=float(exp_config['momentum']),
        weight_decay=float(exp_config['weight_decay'])
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("Loading dataset...")
    dataset = CMNIST()

    # Get the fixed test set
    full_testset = dataset.get_full_testset()

    # loop over all data and compute accuracy after every "batch"
    for i, (x, y, t) in enumerate(dataset):
        print("--------- BATCH {} --------".format(i))

        train_net(
            optimizer, model, criterion, mb_size,
            x, y, y, train_ep
        )

        test_multitask(
            model, full_testset, mb_size, preproc
        )



