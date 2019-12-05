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
Data Loader for the ICifar100 continual learning benchmarks.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import logging
import numpy as np


from .cifar import get_merged_cifar10_and_100, \
    remove_some_labels, read_data_from_pickled


class ICifar100(object):
    """
    iCifar100 (from iCaRL paper) benchmark loader.
    """

    def __init__(self,
                 root_cifar100='/home/admin/data/cifar100/cifar-100-python/',
                 fixed_test_set=False,
                 num_batch=10,
                 cumulative=False,
                 run=0,
                 seed=0,
                 task_sep=True):
        """
        Initialize ICifar100 object.

            Args:
                root_cifar100 (str): location of the Cifar100 Dataset.
                fixed_test_set (bool): if we want a single fixed test set
                                       or one for each batch.
                num_batch (int): The number of training batches.
                cumulative (bool): if we want to accumulate data for every
                                   batch or not.
                run (int): Specific run in case of multi-run execution.
                seed (int): Seed for determistic results.
                task_sep (bool): If we want to consider each batch as separated
                                 tasks or we are in single incremental task.

        """

        self.num_batch = num_batch
        self.classxbatch = 10
        self.tot_num_labels = 100
        self.iter = 0
        self.fixed_test_set = fixed_test_set
        self.cumulative = cumulative
        self.run = run
        self.seed = seed
        self.task_sep = task_sep
        # Getting root logger
        self.log = logging.getLogger('mylogger')

        # load cifar100 images
        self.train_set, self.test_set = read_data_from_pickled(
            root_cifar100, 50000, 10000, 32
        )
        self.all_train_sets = []
        self.all_test_sets = []
        self.tasks_id = []

        print("preparing CL benchmark...")

        # compute which labels select for each batch given run
        labels = list(range(self.tot_num_labels))
        np.random.seed(self.seed)
        if self.run != 0:
            for i in range(self.run):
                np.random.shuffle(labels)

        self.batch2labels = {
            i: labels[i * self.classxbatch:(i + 1) * self.classxbatch]
            for i in range(10)
        }

        for i in range(self.num_batch):

            tr_curr_labels = self.batch2labels[i]
            te_curr_labels = self.batch2labels[i]

            all_labels = range(self.tot_num_labels)

            if self.cumulative:
                # we remove only never seen before labels
                not_remove = []
                for j in range(i + 1):
                    not_remove += self.batch2labels[j]
                tr_labs2remove = [j for j in all_labels if j not in not_remove]
            else:
                # we remove only labels not belonging to the current batch
                tr_labs2remove = [j for j in all_labels if
                                  j not in tr_curr_labels]

            te_labs2remove = [j for j in all_labels if j not in te_curr_labels]

            self.all_train_sets.append(
                remove_some_labels(
                    self.train_set, tr_labs2remove
                )
            )

            if self.fixed_test_set:
                pass
            else:
                self.all_test_sets.append(
                    remove_some_labels(
                        self.test_set, te_labs2remove
                    )
                )

            if self.task_sep:
                self.tasks_id.append(i)
            else:
                self.tasks_id.append(0)

        self.log.debug('Labels order: ' + str(labels) + '\n')

    def __iter__(self):
        return self

    def __next__(self):
        """
        Next batch based on the object parameter which can be also changed
        from the previous iteration.
        """

        if self.iter == self.num_batch:
            raise StopIteration

        train_set = self.all_train_sets[self.iter]

        # get ready for next iter
        self.iter += 1

        return train_set[0], train_set[1], self.tasks_id[self.iter-1]

    def get_grow_test_set(self):
        """
        Return the growing test set (examples up to the current batch).
        """

        # up to num_batches
        all_labels = range(self.tot_num_labels)
        te_curr_labels = []
        for i in range(self.iter+1):
            te_curr_labels += self.batch2labels[i]
        labs2remove = [i for i in all_labels if i not in te_curr_labels]

        return remove_some_labels(self.test_set, labs2remove)

    def get_full_testset(self):
        """
        Return the test set (the same for each inc. batch).
        """
        return list(zip(self.all_test_sets, self.tasks_id))

    next = __next__  # python2.x compatibility.


if __name__ == "__main__":

    # Create the dataset object
    dataset = ICifar100()

    # get test set for each task
    full_testset = dataset.get_full_testset()

    # loop over the training incremental batches
    for i, (x, y, t) in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.

        # do your computation here...
        print("Shapes X: ", x.shape)
        print("Shapes y: ", y.shape)
