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

import ray
from ray import tune

import configparser

from pprint import pprint


class Trainable(tune.Trainable):
    """ray.tune trainable generic class Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(
            self, config=config, logger_creator=logger_creator
        )

    def _setup(self, config):

        # Setting
        self.mb_size = 128
        self.train_ep = 2
        self.use_cuda = True
        self.preproc = None
        self.model = SimpleMLP(
            sparsify=True, percent_on=0.3,
            k_inference_factor=1.5, boost_strength=1.0,
            boost_strength_factor=1, duty_cycle_period=1000,
            num_classes=10, hidden_units=4000
        )
        print(self.model)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, nesterov=True,
            momentum=0.9, weight_decay=1e-4
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        print("Loading dataset...")
        self.dataset = CMNIST()

        # Get the fixed test set
        self.full_testset = self.dataset.get_full_testset()

    def _train(self):

        # loop over all data and compute accuracy after every "batch"
        for i, (x, y, t) in enumerate(self.dataset):
            print("--------- BATCH {} --------".format(i))

            train_net(
                self.optimizer, self.model, self.criterion, self.mb_size,
                x, y, y, self.train_ep
            )

            test_multitask(
                self.model, self.full_testset, self.mb_size, self.preproc
            )
        return {'acc': -1}

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir)
        pass

    def _restore(self, checkpoint):
        self.model.restore(checkpoint)


if __name__ == "__main__":

    import sys
    print(sys.path)

    # setup parameters
    cfg_loc = 'exps/'
    exp_name = 'Exp1'
    config = configparser.ConfigParser()
    config.read(cfg_loc + "exps_params.cfg")

    print(dict(config))
    exp_config = dict(config[exp_name])
    pprint(exp_config)

    tune_config = {
        'resources_per_trial': {
            "cpu": 3,
            "gpu": 0
        },
        "name": exp_name,
        "config": exp_config
    }

    # ray.init(address=args.ray_address)
    # sched = ASHAScheduler(metric="mean_accuracy")
    # analysis = tune.run(
    #     TrainMNIST,
    #     scheduler=sched,
    #     stop={
    #         "mean_accuracy": 0.95,
    #         "training_iteration": 3 if args.smoke_test else 20,
    #     },
    #     resources_per_trial={
    #         "cpu": 3,
    #         "gpu": int(args.use_gpu)
    #     },
    #     num_samples=1 if args.smoke_test else 20,
    #     checkpoint_at_end=True,
    #     checkpoint_freq=3,
    #     config={
    #         "args": args,
    #         "lr": tune.uniform(0.001, 0.1),
    #         "momentum": tune.uniform(0.1, 0.9),
    #     })
    # print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))

    ray.init()

    tune.run(Trainable, **tune_config)



