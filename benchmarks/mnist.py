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
Several utils which can be used for MNIST.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import gzip
import pickle
from copy import deepcopy
from PIL import Image
import os
import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


class MNIST(object):
    """
    MNIST static dataset and basic utilities
    """

    def __init__(self, data_loc='data/'):
        """
        Init method for the MNIST class.

            Args:
                data_loc (str): realtive path in which to download and store
                                mnist data.
        """

        if os.path.isabs(data_loc):
            path = data_loc
        else:
            path = os.path.join(os.path.dirname(__file__), data_loc)

        self.data_loc = path
        self.train_set = None
        self.test_set = None

        try:
            # Create target Directory for MNIST data
            os.mkdir(self.data_loc)
            print("Directory ", self.data_loc, " Created ")
            self.download_mnist()
            self.save_mnist()
            self.load()

        except OSError:
            print("Directory ", self.data_loc, " already exists")
            self.load()

    def load(self):

        with open(self.data_loc + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)

        self.train_set = [mnist["training_images"], mnist["training_labels"]]
        self.test_set = [mnist["test_images"], mnist["test_labels"]]

    def get_data(self):
        """
        Simple method to get train and test set.

            Returns:
                list: train and test sets composed of images and labels.
        """

        return [self.train_set, self.test_set]

    def permute_mnist(self, seed):
        """
        Given the train and test set (no labels), permute pixels of each img
        the same way.

            Args:
                seed (int): seed for the random generator.
            Returns:
                list: train and test images, permuted.
        """

        # we take only the images
        mnist_imgs = [self.train_set[0], self.test_set[0]]

        np.random.seed(seed)
        print("starting permutation...")
        # print(mnist_imgs[0].shape)
        h, w = mnist_imgs[0].shape[2], mnist_imgs[0].shape[3]
        perm_inds = list(range(h*w))
        np.random.shuffle(perm_inds)
        # print(perm_inds)
        perm_mnist = []
        for set in mnist_imgs:
            num_img = set.shape[0]
            # print(num_img, w, h)
            flat_set = set.reshape(num_img, w * h)
            perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, w, h))
        return perm_mnist

    def rotate_mnist(self, rotation):
        """
        Given the train and test set (no labels), rotate each img the
        same way.

            Args:
                rotation (int): degrees of rotation of the images.
            Returns:
                list: rotated train and test images.
        """

        # we take only the images
        mnist_imgs = [self.train_set[0], self.test_set[0]]
        rot_mnist = deepcopy(mnist_imgs)

        for i, set in enumerate(mnist):
            for j in range(set.shape[0]):
                img = Image.fromarray(set[j], mode='L')
                rot_mnist[i][j] = img.rotate(rotation)

        return rot_mnist

    def reduce_mnist(self, classes):
        """
        Given the train and test set (with labels), it returns a subset of it
        with just the classes requested

            Args:
                classes (int): classes to maintain. All the rest will be
                removed.
            Returns:
                list: new dataset composed of the reduced train and test sets.
        """

        idxs_train = np.argwhere(
            np.logical_or(
                self.train_set[1] < classes[0], self.train_set[1] > classes[1]
            )
        )
        idxs_test = np.argwhere(
            np.logical_or(
                self.test_set[1] < classes[0], self.test_set[1] > classes[1]
            )
        )

        new_train_y = np.delete(self.train_set[1], idxs_train)
        new_test_y = np.delete(self.test_set[1], idxs_test)
        new_train_x = np.delete(self.train_set[0], idxs_train, axis=0)
        new_test_x = np.delete(self.test_set[0], idxs_test, axis=0)

        new_dataset = [[new_train_x, new_train_y], [new_test_x, new_test_y]]

        return new_dataset

    def download_mnist(self):
        """
        Download MNIST data from the official website.
        """

        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading " + name[1]+"...")
            urlretrieve(base_url + name[1], self.data_loc + name[1])
        print("Download complete.")

    def save_mnist(self):
        """
        Extract and save MNIST data as a pickled file.
        """

        mnist = {}
        for name in filename[:2]:
            with gzip.open(self.data_loc + name[1], 'rb') as f:
                tmp = np.frombuffer(f.read(), np.uint8, offset=16)
                mnist[name[0]] = tmp.reshape(-1, 1, 28, 28)\
                                     .astype(np.float32) / 255
        for name in filename[-2:]:
            with gzip.open(self.data_loc + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.data_loc + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")


if __name__ == '__main__':

    # np.set_printoptions(threshold=np.nan)
    mnist = MNIST()

    # try permutation
    mnist_data = mnist.get_data()
    perm_mnist = mnist.permute_mnist(seed=0)

    # let's see some images
    print(mnist_data[0][0][0].shape)
    print(perm_mnist[0][0].shape)
