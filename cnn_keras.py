#coding:utf8
import re
import cv2
import os
import numpy as np
import cv2.cv as cv
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
#得到一共多少个样本
def getnum(file_path):
    pathDir = os.listdir(file_path)
    i = 0
    for allDir in pathDir:
        i +=1
    return i
#制作数据集
def data_label(path,count):
    data = np.empty((count,1,128,192),dtype = 'float32')#建立空的四维张量类型32位浮点
    label = np.empty((count,),dtype = 'uint8')
    i = 0
    pathDir = os.listdir(path)
    for each_image in pathDir:
        all_path = os.path.join('%s%s' % (path,each_image))#路径进行连接
        image = cv2.imread(all_path,0)
        mul_num = re.findall(r"\d",all_path)#寻找字符串中的数字，由于图像命名为300.jpg 标签设置为0
        num = int(mul_num[0])-3
#        print num,each_image
#        cv2.imshow("fad",image)
#        print child
        array = np.asarray(image,dtype='float32')
        array -= np.min(array)
        array /= np.max(array)
        data[i,:,:,:] = array
        label[i] = int(num)
        i += 1
    return data,label
#构建卷积神经网络
def cnn_model(train_data,train_label,test_data,test_label):
    model = Sequential()
#卷积层 12 × 120 × 120 大小
    model.add(Convolution2D(
        nb_filter = 12,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'th',
        input_shape = (1,128,192)))
    model.add(Activation('relu'))#激活函数使用修正线性单元
#池化层12 × 60 × 60
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
#卷积层 24 * 58 * 58
    model.add(Convolution2D(
        24,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
#池化层 24×29×29
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
    model.add(Convolution2D(
        48,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides =(2,2),
        border_mode = 'valid'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.4))
    model.add(Dense(5,init = 'normal'))
    model.add(Activation('softmax'))
    adam = Adam(lr = 0.001)
    model.compile(optimizer = adam,
            loss =  'categorical_crossentropy',
            metrics = ['accuracy'])
    print '----------------training-----------------------'
    model.fit(train_data,train_label,batch_size = 20,nb_epoch = 50,shuffle = True,show_accuracy = True,validation_split = 0.1)
    print '----------------testing------------------------'
    loss,accuracy = model.evaluate(test_data,test_label)
    print '\n test loss:',loss
    print '\n test accuracy',accuracy
train_path = '/home/zhanghao/data/classification/train_scale/'
test_path = '/home/zhanghao/data/classification/test_scale/'
train_count = getnum(train_path)
test_count = getnum(test_path)
train_data,train_label = data_label(train_path,train_count)
test_data,test_label = data_label(test_path,test_count)
train_label = np_utils.to_categorical(train_label,nb_classes = 5)
test_label = np_utils.to_categorical(test_label,nb_classes = 5)
cnn_model(train_data,train_label,test_data,test_label)


#print getnum('/home/zhanghao/data/classification/test_scale/')
#data_label('/home/zhanghao/data/classification/test_scale/',1)
#cv.WaitKey(0)
