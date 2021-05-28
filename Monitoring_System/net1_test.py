'''
用于测试是否佩戴口罩，
佩戴口罩为 [1 0 0]
未佩戴口罩为 [0 1 0]
其他情况为 [0 0 1]
'''

import tensorflow as tf
import numpy as np
import load_image_data as lid
import time

def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.maximum(0.1* x, x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def fc(x, w, b):
    x = tf.add(tf.matmul(x, w), b)
    return tf.maximum(0.1* x, x)


def net1(X, weights, biases, drop_out):
    X = tf.reshape(X, [-1, 128, 128, 1])
    conv1 = conv2d(X, weights['conv1'], biases['conv1'])
    pool2 = maxpool2d(conv1)
    conv3 = conv2d(pool2, weights['conv3'], biases['conv3'])
    pool4 = maxpool2d(conv3)
    pool4 = tf.contrib.layers.flatten(pool4)
    fc5 = fc(pool4, weights['fc5'], biases['fc5'])
    fc6 = fc(fc5, weights['fc6'], biases['fc6'])
    return fc6


def load_model(data=None):
    tf.reset_default_graph()
    weights = {
        'conv1': tf.Variable(tf.random_normal([5, 5, 1, 6])),
        'conv3': tf.Variable(tf.random_normal([5, 5, 6, 16])),
        'fc5': tf.Variable(tf.random_normal([16 * 29 * 29, 64])),
        'fc6': tf.Variable(tf.random_normal([64, 3]))
    }
    biases = {
        'conv1': tf.Variable(tf.random_normal([6])),
        'conv3': tf.Variable(tf.random_normal([16])),
        'fc5': tf.Variable(tf.random_normal([64])),
        'fc6': tf.Variable(tf.random_normal([3]))
    }

    X = tf.placeholder(tf.float32, [None, 128 * 128], name='X')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    logits = net1(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        save_path = './model/net1/net1'
        saver.restore(sess, save_path)
        pre = sess.run(prediction, feed_dict={X: data, keep_prob: 1.0})
    return pre


Get_data = lid.BatchLoad()
label = 'D'
indexT, dataT = Get_data.loadtestimg(class_fold='T', label=label)
pre = load_model(dataT)
for k in range(indexT):
    temp = pre[k]
    max_index = np.argmax(temp)
    if max_index == 0:
        result = '已佩戴口罩'
    elif max_index == 1:
        result = '未佩戴口罩!!!!!!'
    else:
        result = '说不清楚'
    print( '样本'+ label + '_%d的识别结果是%s' %(k+1, result))
    time.sleep(0.2)
