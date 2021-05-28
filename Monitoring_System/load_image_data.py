'''
批量载入数据
'''

import cv2
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt


class BatchLoad():
    def __init__(self):
        self.path = './ImageSet/Face_Gray/'

    def loadimg(self, class_fold, label_index):
        newpath = self.path + class_fold + '/'
        filelist = os.listdir(newpath)
        index = 0
        train_data = []
        train_label = []
        for item in filelist:
            if item.endswith('.png'):
                img = cv2.imread(newpath + item)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = gray_img.reshape(128*128)
                train_data.append(img)
                label = np.array([0, 0, 0])
                label[label_index - 1] = 1
                train_label.append(label)
                index += 1

        train_data = np.array(train_data)
        train_label = np.array(train_label)

        return index, train_data, train_label

    def loadtestimg(self, class_fold, label='A'):
        newpath = self.path + class_fold + '/'
        filelist = os.listdir(newpath)
        index = 0
        test_data = []
        for item in filelist:
            if item.endswith('.png') and item.startswith( label + '_'):
                img = cv2.imread(newpath + item)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = gray_img.reshape(128 * 128)
                test_data.append(img)
                index += 1

        test_data = np.array(test_data)
        return index, test_data


# if __name__ == '__main__':
#     demo = BatchLoad()
#     index, train_data, train_label = demo.loadimg(class_fold='A', label_index=1)
#     print(index)