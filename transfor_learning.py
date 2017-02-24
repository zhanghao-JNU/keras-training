import numpy as np
import datetime
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam

now = datetime.datetime.now
batch_size = 128
nb_classes = 5
nb_epoch = 5

img_rows,img_cols = 28,28
nb_filters = 32
pool_size =2
kernel_size = 3
if K.image_dim_ordering()=='th':
    input_shape = (1,img_rows,img_cols)
else:
    input_shape = (img_rows,img_cols,1)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train_lt5 = x_train[y_train <5]
y_train_lt5 = y_train[y_train <5]

x_test_lt5 = x_test[y_test <5]
y_test_lt5 = y_test[y_test <5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5]-5

x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5]-5

def train_model(model,train,test,nb_classes):
    x_train = train[0].reshape((train[0].shape[0],)+input_shape)
    x_test = test[0].reshape((test[0].shape[0],)+ input_shape)
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

    print 'x_train shape',x_train.shape
    print x_train.shape[0],'train samples'
    print x_test.shape[0],'test samples'

    y_train = np_utils.to_categorical(train[1],nb_classes)
    y_test = np_utils.to_categorical(test[1],nb_classes)

    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(),metrics = ['accuracy'])
    t = now()
    model.fit(x_train,y_train,batch_size = batch_size,nb_epoch = nb_epoch,verbose = 1, validation_data = (x_test,y_test))
    print 'train time %s'%(now()-t)
    score = model.evaluate(x_test,y_test,verbose = 0)
    print 'test loss',score[0]
    print 'test accuracy',score[1]
feature_layers = [
        Convolution2D(nb_filters,
            kernel_size,
            kernel_size,
            border_mode = 'valid',
            input_shape = input_shape),
        Activation('relu'),
        Convolution2D(nb_filters,kernel_size,kernel_size),
        Activation('relu'),
        MaxPooling2D(pool_size = (pool_size,pool_size)),
        Dropout(0.25),
        Flatten(),]
classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
        ]

model = Sequential(feature_layers + classification_layers)
train_model(model,(x_train_lt5,y_train_lt5),(x_test_lt5,y_test_lt5),nb_classes)

for L in classification_layers:
    L.feature_layers = False

train_model(model,(x_train_gte5,y_train_gte5),(x_test_gte5,y_test_gte5),nb_classes)

