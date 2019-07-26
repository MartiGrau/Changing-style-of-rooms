# -*- coding: utf-8 -*-
from __future__ import print_function
import chainer
import chainer.functions
import chainer.links.caffe


class NIN(object):
    def __init__(self, caffemodelpath, alpha=[0, 0, 1, 1],
                 beta=[0.000244140625, 6.103515625e-05, 1.52587890625e-05, 3.814697265625e-06], model=None):
        if model is None:
            self.model = chainer.links.caffe.CaffeFunction(caffemodelpath)
        else:
            self.model = model
        self.alpha = alpha
        self.beta = beta

    def forward_layers(self, x, average_pooling=False):
        if average_pooling:
            pooling = lambda x: chainer.functions.average_pooling_2d(chainer.functions.relu(x), 3, stride=2)
        else:
            pooling = lambda x: chainer.functions.max_pooling_2d(chainer.functions.relu(x), 3, stride=2)
        y0 = chainer.functions.relu(self.model.conv1(x))
        y1 = self.model.cccp2(chainer.functions.relu(self.model.cccp1(y0)))
        x1 = chainer.functions.relu(self.model.conv2(pooling(chainer.functions.relu(y1))))
        y2 = self.model.cccp4(chainer.functions.relu(self.model.cccp3(x1)))
        x2 = chainer.functions.relu(self.model.conv3(pooling(chainer.functions.relu(y2))))
        y3 = self.model.cccp6(chainer.functions.relu(self.model.cccp5(x2)))
        x3 = chainer.functions.relu(getattr(self.model, "conv4-1024")(
            chainer.functions.dropout(pooling(chainer.functions.relu(y3)))))
        return [y0, x1, x2, x3]
