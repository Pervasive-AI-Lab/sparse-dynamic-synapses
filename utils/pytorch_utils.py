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

General useful functions for pytorch.

"""
# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable
from .common import pad_data, shuffle_in_unison


def get_accuracy(model, criterion, batch_size, test_x, test_y, test_it,
                 use_cuda=False, mask=None):
    """ Test accuracy given net and data. """

    correct_cnt, ave_loss = 0, 0
    model = maybe_cuda(model, use_cuda=use_cuda)

    num_class = torch.max(test_y) + 1
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class

    for i in range(test_it):
        # indexing
        start = i * batch_size
        end = (i + 1) * batch_size

        x = Variable(
            maybe_cuda(test_x[start:end], use_cuda=use_cuda), volatile=True
        )
        y = Variable(
            maybe_cuda(test_y[start:end], use_cuda=use_cuda), volatile=True
        )

        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.FloatTensor(mask).cuda() == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        loss = criterion(logits, y)
        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()
        ave_loss += loss.data[0]

        for label in y.data:
            pattern_per_class[int(label)] += 1

        for i, pred in enumerate(pred_label):
            if pred == y.data[i]:
                hits_per_class[int(pred)] += 1

    accs = np.asarray(hits_per_class) / \
           np.asarray(pattern_per_class).astype(float)

    acc = correct_cnt * 1.0 / test_y.size(0)

    ave_loss /= test_y.size(0)

    return ave_loss, acc, accs


def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None):
    """ Train net from memory using pytorch """

    cur_ep = 0
    cur_train_t = t

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    for ep in range(train_ep):

        model.active_perc_list = []
        model.train()

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0
        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
            logits = model(x_mb)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

        cur_ep += 1

        print("Average active units: {}%, std {}%: ".format(
              np.mean(model.active_perc_list), np.std(model.active_perc_list))
        )

    return ave_loss, acc


def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """ Here we get a batch of PIL imgs and we return them normalized as for
        the pytorch pre-trained models. """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def maybe_cuda(what, use_cuda=True, **kw):
    """ Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
    """
    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def change_lr(optimizer, lr):
    """Change the learning rate of the optimizer"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_classifier(model, weigth, bias, clas=None):
    """ Change weights and biases of the last layer in the network. """

    if clas is None:
        # model.classifier.weight.data = torch.from_numpy(weigth).float()
        # model.classifier.bias.data = torch.from_numpy(bias).float()
        model.classifier = torch.nn.Linear(512, 10)

    else:
        raise NotImplemented


def reset_classifier(model, val=0, std=None):
    """ Set weights and biases of the last layer in the network to zero. """

    weights = np.zeros_like(model.classifier.weight.data.numpy())
    biases = np.zeros_like(model.classifier.bias.data.numpy())

    if std:
        weights = np.random.normal(
            val, std, model.classifier.weight.data.numpy().shape
        )
    else:
        weights.fill(val)

    biases.fill(0)
    # self.net.classifier[-1].weight.data.normal_(0.0, 0.02)
    # self.net.classifier[-1].bias.data.fill_(0)

    set_classifier(model, weights, biases)


def test_multitask(model, test_set, mb_size, preproc, use_cuda=True):

    model.eval()

    acc_x_task = []
    for (x, y), t in test_set:

        # reduce test set 20 substampling
        # idx = range(0, y.shape[0], 20)
        # x = np.take(x, idx, axis=0)
        # y = np.take(y, idx, axis=0)

        if preproc:
            x = preproc(x)

        (train_x, train_y), it_x_ep = pad_data(
            [x, y], mb_size
        )

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            for it in range(it_x_ep):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()

            acc = correct_cnt.item() / y.shape[0]

        print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)

    print("------------------------------------------")
    print("Avg. acc:", np.mean(acc_x_task))
    print("------------------------------------------")

