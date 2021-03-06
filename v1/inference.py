#!/usr/bin/env python3

import matplotlib.pyplot as plt
import mxnet as mx
import time
import numpy as np
import cv2
from collections import namedtuple

prefix = 'Inception'
epoch = 9
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

if __name__=="__main__":
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    mod = mx.mod.Module(symbol=sym, context=[mx.gpu(0)], label_names=None)
    mod.bind(for_training     = False,
             data_shapes      = [('data', (1,3,224,224))],
             label_shapes = mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    with open('synset.txt', 'r') as f:
      labels = [l.rstrip() for l in f]

    predict('http://writm.com/wp-content/uploads/2016/08/Cat-hd-wallpapers.jpg')
