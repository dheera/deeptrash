#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import PIL
import cv2
from datasets import imagenet

base_name = "mobilenet_v2_1.4_224"
gd = tf.GraphDef.FromString(open(base_name + '_frozen.pb', 'rb').read())

img = np.array(PIL.Image.open('panda.jpg').resize((224, 224))).astype(np.float) / 128 - 1
inp, predictions = tf.import_graph_def(gd,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

cap = cv2.VideoCapture(0)

with tf.Session(graph=inp.graph):
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float) / 128 - 1

        x = predictions.eval(feed_dict={inp: img.reshape(1, 224,224, 3)})

        print(x)
        label_map = imagenet.create_readable_names_for_imagenet_labels()  
        print("Top 1 Prediction: ", x.argmax(),label_map[x.argmax()], x.max())
        cv2.imshow('foo', img)
        cv2.waitKey(1)
