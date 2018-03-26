import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAvgPool2D
from keras.layers import Input, Dense, Activation,BatchNormalization,Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
import keras.callbacks
import cv2
import csv
from keras.utils.np_utils import to_categorical
import codecs
from keras.utils.vis_utils import plot_model
import random

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('shuffle_mats_or_lists only supports '
                            'numpy.array and list objects')
    return ret
class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self,csv_path,picture_path, minibatch_size,
                  val_split,img_w,img_h
                 ):
        self.csv_path = csv_path
        self.picture_path=picture_path
        self.minibatch_size = minibatch_size
        self.cur_train_index = 0
        self.val_split = val_split
        self.cur_val_index = val_split
        self.img_w = img_w
        self.img_h = img_h


        csvfile = file(self.csv_path, 'rb')
        reader = csv.reader(csvfile)
        self.img_path = []
        self.all_label = []
        for line in reader:
            cnt = 1000
            for i in range(50):
                self.img_path.append(self.picture_path +line[0].split('.')[0] + '_%d' % cnt + '.bmp')
                self.all_label.append(float(line[1])/100)
                cnt = cnt + 1
        csvfile.close()
        self.img_path,self.all_label = shuffle_mats_or_lists([self.img_path,self.all_label])
    def get_batch(self, index, size):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            self.X_data = np.ones([size,3,self.img_w,self.img_h])
        else:
            self.X_data = np.ones([size,self.img_w,self.img_h,3])
        self.label = np.ones(size)
        self.labe11 = np.empty([size,10])
        for i in range(size):
            categorical_labels = to_categorical(1, num_classes=10)
            self.labe11[i,:] = categorical_labels
        self.batch_path = []
        for i in range(0,size):
            if K.image_data_format()=='channels_first':
                image = cv2.imread(self.img_path[index+i])
                array = np.asarray(image, dtype='float32')
                array -= np.min(array)
                array /= (np.max(array) - np.min(array))
                array = array.transpose(0,2)
                array = array.transpose(1, 2)
                array = np.expand_dims(array, axis=0)
                self.X_data[i,:,:,:] = array

                #print ('channels_first'),
                #print X_data.shape
            else:
                image = cv2.imread(self.img_path[index + i])
                array = np.asarray(image, dtype='float32')
                #print array.shape
                array -= np.min(array)
                array /= (np.max(array) - np.min(array))
                array = np.expand_dims(array, axis=0)
                self.X_data[i, :, :, :] = array
            self.label[i] = self.all_label[index + i]
            self.batch_path.append(self.img_path[index + i])
        #return self.X_data,self.label
        return (self.X_data, [self.label,self.labe11])
    def next_train(self):
        while 1:
            #print ('next_train.....')
            ret = self.get_batch(self.cur_train_index, self.minibatch_size)
            self.cur_train_index += self.minibatch_size
            #print self.cur_train_index
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % self.minibatch_size
                (self.img_path[:self.val_split],self.all_label[:self.val_split]) = shuffle_mats_or_lists\
                    ([self.img_path[:self.val_split],self.all_label[:self.val_split]])
            yield ret
    def next_val(self):
        while 1:
            #print ('next_val.....')
            ret = self.get_batch(self.cur_val_index, self.minibatch_size)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= len(self.all_label):
                self.cur_val_index = self.val_split + self.cur_val_index % self.minibatch_size
            yield ret


def my_net(img_w,img_h,n_channels):
    x = Input(shape=(img_w,img_h,n_channels))
    conv1_1 = Conv2D(filters=32,kernel_size=(3,1),strides=(1,1),padding='same',activation='relu')(x)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    bn1_1 = BatchNormalization(axis=3)(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(bn1_1)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(pool1_1)
    conv1_4 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_3)
    bn1_2 = BatchNormalization(axis=3)(conv1_4)
    pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn1_2)
    conv1_5 = Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(pool1_2)
    conv1_6 = Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_5)
    bn1_3 = BatchNormalization(axis=3)(conv1_6)
    pool1_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn1_3)
    conv1_7 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(pool1_3)
    conv1_8 = Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1_7)
    bn1_4 = BatchNormalization(axis=3)(conv1_8)
    pool1_4 = GlobalAvgPool2D(data_format='channels_last')(bn1_4)

    conv3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    bn3_1 = BatchNormalization(axis=3)(conv3_2)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(bn3_1)
    conv3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_1)
    conv3_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_3)
    bn3_2 = BatchNormalization(axis=3)(conv3_4)
    pool3_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn3_2)
    conv3_5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_2)
    conv3_6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_5)
    bn3_3 = BatchNormalization(axis=3)(conv3_6)
    pool3_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(bn3_3)
    conv3_7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_3)
    conv3_8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_7)
    bn3_4 = BatchNormalization(axis=3)(conv3_8)
    pool3_4 = GlobalAvgPool2D(data_format='channels_last')(bn3_4)

    out = Concatenate(axis=1)([pool1_4,pool3_4])
    break_out = Dense(units=1000,activation='relu')(out)
    out1 = Dense(units=1,activation='sigmoid',name='out1')(break_out)
    out2 = Dense(units=10,activation='softmax',name='out2')(break_out)
    model = Model(inputs=x,outputs=[out1,out2])
    model.compile(optimizer=Adam(lr=0.001),
                  loss={'out1': 'mse', 'out2': 'categorical_crossentropy'},
                  loss_weights={'out1': 1., 'out2': 0.2})
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
    all_sample = 10050

    val_split = 9050
    minibatch_size = 5
    img_gen = TextImageGenerator("./train.csv", "./train_patch/", minibatch_size=minibatch_size, val_split=val_split, img_w=256, img_h=256)
    model.fit_generator(generator=img_gen.next_train(),
                         samples_per_epoch=(val_split / minibatch_size)
                         , nb_epoch=200, callbacks=[ModelCheckpoint('./my_net.h5',
                                                                                        monitor='val_loss',
                                                                                        verbose=1,
                                                                                        save_best_only=True,
                                                                                        save_weights_only=True,
                                                                                        mode='auto',
                                                                                        period=1)],
                         validation_data=img_gen.next_val(), nb_val_samples=((all_sample-val_split) / minibatch_size))
    return model

if __name__ == '__main__':
    model = my_net(256,256,3)
    model.summary()
