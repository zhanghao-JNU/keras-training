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
batch_size = 20
nb_epoch = 500
x_pro = preprocessing.MinMaxScaler(feature_range = (0,1))
x_train = x_pro.fit_transform(x_train)
x_test = x_pro.transform(x_test)
y_pro = preprocessing.MinMaxScaler(feature_range = (0,1))
y_train = y_pro.fit_transform(y_train.reshape(-1,1))
y_test = y_pro.transform(y_test.reshape(-1,1))
model = Sequential()
model.add(Dense(50,activation = 'relu',input_dim = 13))
model.add(Dropout(0.4))
#model.add(Dense(20,activation = 'relu'))
#model.add(Dropout(0.4))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()

adam = Adam(lr = 0.00005)
model.compile(optimizer = adam,loss = 'mse',metrics = ['accuracy'])
model.fit(x_train,y_train,nb_epoch = nb_epoch,batch_size = batch_size,shuffle = True,validation_data = (x_test,y_test))

pred = model.predict(x_test)
l = len(y_test)
plt.plot(range(l),y_test,'r',lw = 2)
plt.plot(range(l),pred,'b',lw = 2)
plt.show()
