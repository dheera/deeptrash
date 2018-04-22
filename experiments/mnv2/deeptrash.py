#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import PIL
import cv2
from datasets import imagenet

GRAPH_FILENAME = "mobilenet_v2_1.4_224_frozen.pb"
INPUT_NODE = "input:0"
OUTPUT_NODE = "MobilenetV2/Predictions/Reshape_1:0"

gd = tf.GraphDef.FromString(open(GRAPH_FILENAME, 'rb').read())

input_tensor, predictions_tensor = tf.import_graph_def(gd,  return_elements = [INPUT_NODE, OUTPUT_NODE])

cap = cv2.VideoCapture(0)
label_map = imagenet.create_readable_names_for_imagenet_labels()  

with tf.Session(graph=input_tensor.graph):
    while True:
        ret, img = cap.read()
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_NEAREST).astype(np.float) / 128 - 1

        t = time.time()
        x = predictions_tensor.eval(feed_dict={input_tensor: img.reshape(1, 224,224, 3)})
        print("inference time: " + str(time.time() - t))

        print("Top 1 Prediction: ", x.argmax(),label_map[x.argmax()], x.max())
        cv2.imshow('foo', img)
        cv2.waitKey(1)
