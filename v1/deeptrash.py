#!/usr/bin/env python3

import matplotlib.pyplot as plt
import mxnet as mx
import time
import numpy as np
import cv2
import json
from collections import namedtuple
from nltk.corpus import wordnet

prefix = 'Inception'
epoch = 9
Batch = namedtuple('Batch', ['data'])

cap = cv2.VideoCapture(0)

def predict(img):
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5

    prob = np.squeeze(prob)
    for ignore_index in trash_indexes['_ignore']:
         prob[ignore_index] = 0.0

    a = np.argsort(prob)[::-1]

    trash_category_probs = [ 0.0 ] * len(trash_categories)

    for i in a[0:5]:
        if trash_revindex[i] != -1:
            trash_category_probs[trash_revindex[i]] += prob[i]

    for i, category in enumerate(trash_categories):
        print(category, trash_category_probs[i])
         

if __name__=="__main__":
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    mod = mx.mod.Module(symbol=sym, context=[mx.gpu(0)], label_names=None)
    mod.bind(for_training     = False,
             data_shapes      = [('data', (1,3,224,224))],
             label_shapes = mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    trash_revindex = [ -1 ] * 32768

    with open('trash_indexes.json', 'r') as f:
      trash_indexes = json.loads(f.read())

    trash_categories = list(trash_indexes.keys())
    for i in range(len(trash_categories)):
         for index in trash_indexes[trash_categories[i]]:
              trash_revindex[index] = i

    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) #, interpolation = cv2.INTER_NEAREST) #.astype(np.float) / 128 - 1

        t = time.time()
        predict(img)
        print("inference time: " + str(time.time() - t))

        cv2.imshow('foo', img)
        cv2.waitKey(1)
