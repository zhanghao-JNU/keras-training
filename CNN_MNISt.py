import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Activation,Flatten,Convolution2D,MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

batch_size = 128
nb_classes = 10
nb_epoch = 12
img_rows,img_cols = 28,28
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)

(X_train,y_train),(X_test,y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
    X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
    X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#print X_test.shape

Y_trian = np_utils.to_categorical(y_train,nb_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes = nb_classes)

model = Sequential()
model.add(Convolution2D(
    nb_filters,
    kernel_size[0],
    kernel_size[1],
    border_mode = 'same',
    input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size = pool_size,
    strides = (2,2)))
model.add(Convolution2D(
    nb_filters,
    kernel_size[0],
    kernel_size[1],
    border_mode = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size = pool_size,
    strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#model.summary()
#plot(model,to_file = 'model-CNN.png')
adam = Adam(lr = 0.0001)
model.compile(loss = 'categorical_crossentropy',
        optimizer = adam,
        metrics = ['accuracy'])
model.fit(X_train,Y_trian,batch_size = batch_size,nb_epoch = nb_epoch,
        validation_split = 0.2,shuffle = True)
score = model.evaluate(X_test,Y_test)
print 'test loss ',score[0]
print 'test accuracy',score[1]
