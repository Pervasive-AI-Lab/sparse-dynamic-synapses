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
Single entry point script to run the experiments of the project. It takes in
input an experiment configuration (DEFAULT is non is provided) described in
the exps/exps_params.cfg file.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from models.simple_mlp import SimpleMLP
from models.simple_cnn import SimpleCNN
from benchmarks.cmnist import CMNIST
from benchmarks.ccifar import ICifar100
from utils.pytorch_utils import train_net, test_multitask, preprocess_imgs

import torch
import configparser
from pprint import pprint
import pickle as pkl
import datetime
import copy
import os
import time
import argparse

if __name__ == "__main__":

    # recover exp configuration name
    parser = argparse.ArgumentParser(description='Run CL experiments')
    parser.add_argument('--name', dest='exp_name',  default='DEFAULT',
                        help='name of the experiment you want to run.')
    args = parser.parse_args()

    # set cuda device (based on your hardware)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # recover config file for the experiment
    cfg_loc = 'exps/'
    config = configparser.ConfigParser()
    config.read(cfg_loc + "exps_params.cfg")
    exp_config = config[args.exp_name]
    print("Expetiment name:", args.exp_name)
    pprint(dict(exp_config))

    # recover parameters from the cfg file and compute the dependent ones.
    mb_size = int(exp_config['mb_size'])
    train_ep = int(exp_config['train_ep'])
    train_ep_inc = int(exp_config['train_ep_inc'])
    cnn = exp_config.getboolean('cnn')
    nesterov = exp_config.getboolean('nesterov')
    cumul = exp_config.getboolean('cumul')
    record_stats = exp_config.getboolean('record_stats')

    if exp_config['benchmark'] == 'mnist':
        input_size = 28 * 28
        image_size = 28
        num_classes = 10
        channels = 1
        preproc = None
    else:
        input_size = 32 * 32 * 3
        image_size = 32
        channels = 3
        num_classes = 100
        preproc = preprocess_imgs

    use_cuda = True
    stats = {
        'perc_on_avg': [],
        'perc_on_std': [],
        'accs': [],
        'on_idxs': []
    }

    # we keep heads in case of multi-head approach (heads are essentially
    # snapshot of the last fc layer at the end of the training for each task.
    # Please note that we do this only for the SPlit MNIST and ICifar100
    # benchmarks.
    heads = []

    # creating the model
    if cnn:
        model = SimpleCNN(
            sparsify=exp_config.getboolean('sparsify'),
            percent_on_fc=float(exp_config['percent_on_fc']),
            percent_on_conv=float(exp_config['percent_on_conv']),
            k_inference_factor=float(exp_config['k_inference_factor']),
            boost_strength=float(exp_config['boost_strength']),
            boost_strength_factor=float(exp_config['boost_strength_factor']),
            duty_cycle_period=int(exp_config['duty_cycle_period']),
            num_classes=num_classes,
            hidden_units=int(exp_config['hidden_units']),
            dropout=float(exp_config['dropout']),
            weight_sparsity_fc=float(exp_config['weight_sparsity_fc']),
            weight_sparsity_conv=float(exp_config['weight_sparsity_conv']),
            image_size=image_size,
            channels=channels,
            stats=record_stats
        )
    else:
        model = SimpleMLP(
            sparsify=exp_config.getboolean('sparsify'),
            percent_on=float(exp_config['percent_on_fc']),
            k_inference_factor=float(exp_config['k_inference_factor']),
            boost_strength=float(exp_config['boost_strength']),
            boost_strength_factor=float(exp_config['boost_strength_factor']),
            duty_cycle_period=int(exp_config['duty_cycle_period']),
            num_classes=num_classes,
            hidden_layers=int(exp_config['hidden_layers']),
            hidden_units=int(exp_config['hidden_units']),
            dropout=float(exp_config['dropout']),
            weight_sparsity=float(exp_config['weight_sparsity_fc']),
            input_size=input_size,
            stats=record_stats
        )

    # printing description of the model used
    print(model)

    # setting up optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(exp_config['lr']),
        nesterov=nesterov,
        momentum=float(exp_config['momentum']),
        weight_decay=float(exp_config['weight_decay'])
    )
    criterion = torch.nn.CrossEntropyLoss()

    # loading dataset
    print("Loading dataset...")
    if exp_config['benchmark'] == 'mnist':
        kwargs = {
            'num_batch': int(exp_config['num_batch']),
            'mode': exp_config['mnist_mode']
        }
        dataset = CMNIST(**kwargs)

    elif exp_config['benchmark'] == 'cifar':
        kwargs = {
            'num_batch': int(exp_config['num_batch'])
        }
        dataset = ICifar100(**kwargs)
    else:
        raise NotImplemented

    # Get the fixed test set
    full_testset = dataset.get_full_testset()

    # start timer
    start_time = time.time()

    if cumul:
        # in this case the training is cumulative (i.e. one shot on the entire
        # training set).
        _, _, cur_stats = train_net(
            optimizer, model, criterion, mb_size,
            dataset.train_set[0], dataset.train_set[1], 0, train_ep,
            preproc=preproc, record_stats=record_stats
        )

        for k, new_seq in cur_stats.items():
            if 'on_idxs' == k:
                stats[k].append(new_seq)
            else:
                stats[k] += new_seq

        cur_stats = test_multitask(
            model, full_testset, mb_size, mask=True, preproc=preproc
        )

        stats['accs'].append(cur_stats['accs'])

    else:
        # loop over all data and compute accuracy after every "batch/task"
        for i, (x, y, t) in enumerate(dataset):
            print("--------- BATCH {} --------".format(i))

            if i == 1:
                train_ep = train_ep_inc

            # if i < 2: continue

            _, _, cur_stats = train_net(
                optimizer, model, criterion, mb_size,
                x, y, y, train_ep, preproc=preproc, record_stats=record_stats
            )

            if exp_config['mnist_mode'] == "split" or \
                    exp_config['benchmark'] == 'cifar':
                # then we use a multi-head approach
                heads.append(copy.deepcopy(model.classifier))

            for k, new_seq in cur_stats.items():
                if 'on_idxs' == k:
                    stats[k].append(new_seq)
                else:
                    stats[k] += new_seq

            cur_stats = test_multitask(
                model, full_testset, mb_size, multi_heads=heads,
                preproc=preproc
            )

            stats['accs'].append(cur_stats['accs'])

    # stop timer
    elapsed_time = time.time() - start_time
    print("Elapsed time (m): ", elapsed_time / 60)

    if record_stats:
        # save results in picked file in results/
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open('results/' + args.exp_name + '_' +
                  time_str + '.pkl', 'wb') as wf:
            pkl.dump([exp_config, stats], wf)



