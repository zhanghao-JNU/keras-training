import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy

import tensorflow as tf
from keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 20

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

print x_train.shape[0],'train samples'
print x_test.shape[0],'test samples'

y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

model = Sequential()
model.add(Dense(512,input_shape = (784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',
        optimizer = Adam(),
        metrics = ['accuracy'])
history = model.fit(x_train,y_train,batch_size = batch_size,nb_epoch = nb_epoch,validation_data = (x_test,y_test))

score = model.evaluate(x_test,y_test)
print 'test loss',score[0]
print 'test accuracy',score[1]
