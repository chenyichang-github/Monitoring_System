'''
用于训练是否佩戴口罩，
佩戴口罩为 [1 0 0]
未佩戴口罩为 [0 1 0]
其他情况为 [0 0 1]
'''
import tensorflow as tf
import numpy as np
import scipy.io as scio
import time
import matplotlib.pyplot as plt
import gc
import load_image_data as lid


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


weights = {
    'conv1': tf.Variable(tf.random_normal([5, 5, 1, 6])),
    'conv3': tf.Variable(tf.random_normal([5, 5, 6, 16])),
    'fc5': tf.Variable(tf.random_normal([16*29*29, 64])),
    'fc6': tf.Variable(tf.random_normal([64, 3]))
}

biases = {
    'conv1': tf.Variable(tf.random_normal([6])),
    'conv3': tf.Variable(tf.random_normal([16])),
    'fc5': tf.Variable(tf.random_normal([64])),
    'fc6': tf.Variable(tf.random_normal([3]))
}


# load_data_train = scio.loadmat('./ImageSet/Face_Gray/A/data_train_A.mat')
# load_label_train = scio.loadmat('./ImageSet/Face_Gray/A/label_train_A_net1.mat')
# dataA = load_data_train.get('data_train_A')
# labelA = load_label_train.get('label_train_A_net1')
# load_data_train = scio.loadmat('./ImageSet/Face_Gray/B/data_train_B.mat')
# load_label_train = scio.loadmat('./ImageSet/Face_Gray/B/label_train_B_net1.mat')
# dataB = load_data_train.get('data_train_B')
# labelB = load_label_train.get('label_train_B_net1')
# load_data_train = scio.loadmat('./ImageSet/Face_Gray/C/data_train_C.mat')
# load_label_train = scio.loadmat('./ImageSet/Face_Gray/C/label_train_C_net1.mat')
# dataC = load_data_train.get('data_train_C')
# labelC = load_label_train.get('label_train_C_net1')
# load_data_train = scio.loadmat('./ImageSet/Face_Gray/D/data_train_D.mat')
# load_label_train = scio.loadmat('./ImageSet/Face_Gray/D/label_train_D_net1.mat')
# dataD = load_data_train.get('data_train_D')
# labelD = load_label_train.get('label_train_D_net1')
# load_data_train = scio.loadmat('./ImageSet/Face_Gray/F/data_train_F.mat')
# load_label_train = scio.loadmat('./ImageSet/Face_Gray/F/label_train_F_net1.mat')
# dataF = load_data_train.get('data_train_F')
# labelF = load_label_train.get('label_train_F_net1')
# data = np.append(dataA, dataB, axis=0)
# data = np.append(data, dataC, axis=0)
# data = np.append(data, dataD, axis=0)
# data = np.append(data, dataF, axis=0)
# label = np.append(labelA, labelB, axis=0)
# label = np.append(label, labelC, axis=0)
# label = np.append(label, labelD, axis=0)
# label = np.append(label, labelF, axis=0)
# del dataA, labelA, dataB, labelB, dataC, labelC, dataD, labelD, dataF, labelF
# gc.collect()

Get_data = lid.BatchLoad()
indexA, dataA, labelA = Get_data.loadimg(class_fold='A', label_index=1)
indexB, dataB, labelB = Get_data.loadimg(class_fold='B', label_index=2)
indexC, dataC, labelC = Get_data.loadimg(class_fold='C', label_index=1)
indexD, dataD, labelD = Get_data.loadimg(class_fold='D', label_index=2)
indexF, dataF, labelF = Get_data.loadimg(class_fold='F', label_index=3)
data = np.append(dataA, dataB, axis=0)
data = np.append(data, dataC, axis=0)
data = np.append(data, dataD, axis=0)
data = np.append(data, dataF, axis=0)
label = np.append(labelA, labelB, axis=0)
label = np.append(label, labelC, axis=0)
label = np.append(label, labelD, axis=0)
label = np.append(label, labelF, axis=0)
del dataA, labelA, dataB, labelB, dataC, labelC, dataD, labelD, dataF, labelF
index_num = indexA + indexB + indexC + indexD + indexF
del indexA, indexB, indexC, indexD, indexF
gc.collect()



picture_num = data.shape[0]
batch_size = 193
learning_rate = 1e-3
display_step = 1
test_step = 1
num_steps = 100
dropout = 0.5
l2_lambda = 1e-5

X = tf.placeholder(tf.float32, [None, 128*128], name='X')
Y = tf.placeholder(tf.float32, [None, 3], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
logits = net1(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(l2_lambda),
                                                 weights_list=tf.trainable_variables())
final_loss = loss_op + l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(final_loss)
init = tf.global_variables_initializer()
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
data_train_num = int(picture_num/1)
train_accuracy = list()
train_loss = list()
cycle_index = int(data_train_num/batch_size)

start_time = time.clock()
with tf.Session() as sess:
    sess.run(init)
    global_train = 0
    data_train = data
    label_train = label
    batch_data_index = np.random.permutation(range(data_train_num))
    for step in range(1, num_steps + 1):
        for i in range(cycle_index):
            batch_data = data_train[batch_data_index[batch_size * i:(batch_size * (i + 1))]]
            batch_label = label_train[batch_data_index[batch_size * i:(batch_size * (i + 1))]]
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                pre, loss, acc = sess.run([prediction, loss_op, accuracy],
                                          feed_dict={X: batch_data, Y: batch_label, keep_prob: 1.0})
                train_loss.append(loss)
                train_accuracy.append(acc)
                print("Step" + str((step - 1) * cycle_index + i + 1) + \
                      ", Minibatch Loss= " + "{:.4f}".format(loss) + \
                      ", Training Accuracy= " + "{:.3f}".format(acc))

    end_time = time.clock()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    save_path = './model/net1/net1'
    saver.save(sess, save_path)
    plt.plot(train_loss)
    plt.plot(train_accuracy)
    plt.show()
    print(end_time - start_time)
    print("Optimization Finished!")
str1 = 'train_loss'
str2 = 'train_accuracy'
dataNew1 = './model/loss_mat/train_loss_net1.mat'
dataNew2 = './model/loss_mat/train_accuracy_net1.mat'
scio.savemat(dataNew1, {str1: train_loss})
scio.savemat(dataNew2, {str2: train_accuracy})




