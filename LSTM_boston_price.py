from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import Adam
boston = load_boston()
x = boston.data
y = boston.target
l = len(x)
#plt.scatter(range(l),y)
#plt.show()
#print y.shape
x_train = x[:450,:]
y_train = y[:450]
x_test = x[450:,:]
y_test = y[450:]
#print y_test.shape
#l = len(x_train)
#plt.scatter(range(l),y_train)
x_pro = preprocessing.MinMaxScaler(feature_range = (-1,1))
x_train = x_pro.fit_transform(x_train)
x_test = x_pro.transform(x_test)
y_pro = preprocessing.MinMaxScaler(feature_range = (-1,1))
y_train = y_pro.fit_transform(y_train.reshape(-1,1))
y_test = y_pro.transform(y_test.reshape(-1,1))
x_train = x_train.reshape(-1,1,13)
x_test = x_test.reshape(-1,1,13)
#print x_train.shape,x_test.shape
#l = len(x_train)
#plt.scatter(range(l),y_train)
#plt.show()

batch_size = 12
nb_epoch = 1000

rows,cols = 1,13
output_size = 1

nb_lstm_outputs = 20
nb_time_steps = rows
dim_input_vector = cols
input_shape = (nb_time_steps,dim_input_vector)
model = Sequential()
model.add(LSTM(nb_lstm_outputs,input_shape = input_shape))
#model.add(LSTM(nb_lstm_outputs,batch_input_shape = (batch_size,rows,cols),stateful = True))
#model.add(Dense(10,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(output_size,activation = 'tanh'))
model.summary()

adam = Adam(lr = 0.0001)
model.compile(optimizer = adam,loss = 'mse',metrics = ['accuracy'])
model.fit(x_train,y_train,nb_epoch = nb_epoch,batch_size = batch_size,shuffle = True,validation_data = (x_test,y_test))

pred = model.predict(x_test)
l = len(y_test)
plt.plot(range(l),y_test,'r',lw = 2)
plt.plot(range(l),pred,'b',lw = 2)
plt.show()
