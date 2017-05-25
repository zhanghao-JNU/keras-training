import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,BatchNormalization
from keras.layers import Reshape, Lambda,Dropout,merge
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
import keras.callbacks
import cv2
import codecs
import count
INDEX_TABEL = count.GenIndex()
np.random.seed(55)
img_w=384
img_h = 42
words_per_epoch = 102400
val_split = 0.2
minibatch_size =32
val_words = int(words_per_epoch * (val_split))
id_picture_path = '/home/zhanghao/pycharm_object/1/name/'
id_txt_path = '/home/zhanghao/pycharm_object/1/shuffle_name.txt'
def show_table():
    for data in INDEX_TABEL:
        print data,
def text_to_labels(word):
    ret = []
    for each in word:
        ret.append(INDEX_TABEL.index(each))
    return ret
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

    def __init__(self,id_picture_path,id_txt_path, minibatch_size,
                 img_w, img_h, val_split,
                 absolute_max_string_len=16):
        self.id_picture_path = id_picture_path
        self.id_txt_path = id_txt_path
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.val_split = val_split
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(INDEX_TABEL)+1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words):
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = []
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.X_text_path = []
        self.Y_len = [0] * self.num_words
        with codecs.open(self.id_txt_path,'r','utf-8') as f:
            chinese = f.readlines()
        for line in chinese:
            word = line.rstrip()
            self.string_list.append(word)
        for i,word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i,0:len(word)] = text_to_labels(word)
            self.X_text.append(word)
        pathDir = os.listdir(self.id_picture_path)
        pathDir = sorted(pathDir)
        for each_path in pathDir:
            all_path = os.path.join('%s%s' % (self.id_picture_path, each_path))
            self.X_text_path.append(all_path)
        #print len(self.X_text_path)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
        self.cur_val_index = self.val_split
        #print (self.cur_val_index)
        self.cur_train_index = 0
        #print ('build done......................')
    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size,1,self.img_w,self.img_h])
        else:
            X_data = np.ones([size,self.img_w,self.img_h,1])
        labels = np.ones([size,self.absolute_max_string_len])
        input_length = np.zeros([size,1])
        label_length = np.zeros([size,1])
        source_str = []
        for i in range(0,size):
            if K.image_data_format()=='channels_first':
                image = cv2.imread(self.X_text_path[index+i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                array = np.asarray(image, dtype='float32')
                array -= np.min(array)
                array /= (np.max(array) - np.min(array))
                array = array.transpose(1,0)
                array = np.expand_dims(array, axis=0)
                array = np.expand_dims(array, axis=0)
                X_data[i,:,:,:] = array
                #print ('channels_first'),
                #print X_data.shape
            else:
                image = cv2.imread(self.X_text_path[index + i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                array = np.asarray(image, dtype='float32')
                array -= np.min(array)
                array /= (np.max(array) - np.min(array))
                array = array.transpose(1,0)
                array = np.expand_dims(array, axis=2)
                array = np.expand_dims(array, axis=0)
                X_data[i, :, :, :] = array
                #print self.X_text_path[index+i]
                #int ('channels_later'),
                #print X_data.shape
            labels[i, :] = self.Y_data[index + i]
            input_length[i] = self.img_w // 8-2
            label_length[i] = self.Y_len[index + i]
            source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  #'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        #print ('return data')
        #return (inputs, outputs)
        return [X_data, labels, input_length, label_length], np.zeros([size])
    def next_train(self):
        while 1:
            #print ('next_train.....')
            ret = self.get_batch(self.cur_train_index, self.minibatch_size)
            self.cur_train_index += self.minibatch_size
            #print self.cur_train_index
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % self.minibatch_size
                (self.X_text_path, self.Y_data, self.Y_len,self.X_text) = shuffle_mats_or_lists(
                   [self.X_text_path, self.Y_data, self.Y_len,self.X_text], self.val_split)
            yield ret
    def next_val(self):
        while 1:
            #print ('next_val.....')
            ret = self.get_batch(self.cur_val_index, self.minibatch_size)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % self.minibatch_size
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(words_per_epoch)


def ctc_lambda_func(args):
    y_pred,labels,input_length,label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels,y_pred,input_length,label_length)
def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == -1:
            break
        ret.append(int(l[i]))
    return ret
def evaluate(base_model,batch_num = 32):
    img_gen = TextImageGenerator(id_picture_path, id_txt_path,
                                 minibatch_size=batch_num,
                                 img_w=img_w,
                                 img_h=img_h,
                                 val_split=words_per_epoch - val_words,
                                 absolute_max_string_len=6
                                 )
    img_gen.build_word_list(words_per_epoch)
    generator = img_gen.next_val()
    [X_test, y_test, _, _], _ = next(generator)
    y_pred = base_model.predict(X_test)
    shape = y_pred[:, 2:, :].shape
    out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :]
    num = 0
    words = 0
    all_num = 0
    for i in range(batch_num):
        y_test1 = remove_blank(y_test[i])
        out1 = remove_blank(out[i])
        all_num = len(y_test1)+all_num
        #print y_test1
        #print out1
        if (len(y_test1) == len(out1)):
            match =True
            if y_test1 != out1:
                match= False
            if (match):
                words = words+1
        if (len(y_test1)>len(out1)):
            for i in range(len(out1)):
                if y_test1[i] == out1[i]:
                    num = num+1
        else:
            for i in range(len(y_test1)):
                if y_test1[i] == out1[i]:
                    num = num+1
    print num,all_num
    return words / float(batch_num),num/float(all_num)
global base_model
class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.accs = []
        self.num_acc = []
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model,32)
        words_acc = acc[0] *100
        num_acc =acc[1] *100
        #self.accs.append(words_acc)
        #self.num_acc.append(num_acc)
        print
        print 'acc: %f%%' % words_acc
        print 'num_acc:%f%%'%num_acc
def cnn_lstm_ctc(run_name, start_epoch, stop_epoch):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
        axis = 1
    else:
        input_shape = (img_w, img_h,1)
        axis = 3
    img_gen = TextImageGenerator(id_picture_path, id_txt_path,
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 val_split= words_per_epoch - val_words,
                                 absolute_max_string_len = 6
                                 )
    input_date = Input(shape=input_shape,name='input_data',dtype='float32')
    out = Conv2D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu',name='Conv1_1')(input_date)
    out = Conv2D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu',name='Conv1_2')(out)
    out = MaxPooling2D(pool_size=(2,2),strides=2,name='Maxpool1')(out)
    out = Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu',name='Conv2_1')(out)
    out = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv2_2')(out)
    out = MaxPooling2D(pool_size=(2, 2), strides=2, name='Maxpool2')(out)
    out = Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu',name='Conv3_1')(out)
    out = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='Conv3_2')(out)
    out = MaxPooling2D(pool_size=(2, 2), strides=2, name='Maxpool3')(out)
    out = BatchNormalization(axis=axis)(out)
    conv_shape = out.get_shape()
    conv_to_lstm_dim = (int(conv_shape[1]),int(conv_shape[2]*conv_shape[3]))
    out = Reshape(target_shape=conv_to_lstm_dim)(out)
    # gru_1 = GRU(256, return_sequences=True, init='he_normal', name='gru1')(out)
    # gru_1b = GRU(256, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(out)
    # gru1_merged = merge([gru_1, gru_1b], mode='sum')
    #
    # gru_2 = GRU(256, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    # gru_2b = GRU(256, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
    lstm1 = Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True, name='lstm1'), merge_mode='sum')(
        out)
    lstm2 = Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True, name='lstm2'))(lstm1)
    #out = merge([gru_2, gru_2b], mode='concat')
    out = Dropout(0.2)(lstm2)
    y_pred = Dense(units= img_gen.get_output_size(),activation='softmax')(out)
    global base_model
    base_model = Model(inputs=input_date,outputs=y_pred)
    base_model.summary()
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(input=[input_date, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=0.001))
    evaluator = Evaluate()
    # model.fit_generator(generator=img_gen.next_train(),
    #                      samples_per_epoch=((words_per_epoch-val_words)/minibatch_size)
    #                      , nb_epoch=200,callbacks=[img_gen, evaluator],
    #                      validation_data=img_gen.next_val(), nb_val_samples=(val_words/minibatch_size))

    #base_model.summary()
if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    cnn_lstm_ctc(run_name=run_name, start_epoch=0, stop_epoch=20)
#     id_picture_path = '/home/zhanghao/pycharm_object/1/name/'
#     id_txt_path = '/home/zhanghao/pycharm_object/1/shuffle_name.txt'
#     img_gen = TextImageGenerator(id_picture_path, id_txt_path,
#                                  minibatch_size=minibatch_size,
#                                  img_w=img_w,
#                                  img_h=img_h,
#                                  val_split= words_per_epoch - val_words,
#                                  absolute_max_string_len = 6
#                                  )
#     img_gen.build_word_list(words_per_epoch)
#     data = img_gen.next_train()
#     print next(data)