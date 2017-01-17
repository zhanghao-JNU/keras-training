import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam,SGD
X = np.linspace(1,20,1000)
X = X[:,np.newaxis]
y = np.sin(X) + np.random.normal(0,0.08,(1000,1))
min_max_scaler = MinMaxScaler((0,1))
y_train = min_max_scaler.fit_transform(y)
x_train = min_max_scaler.fit_transform(X)
#print X.shape ,y.shape
#np.random.shuffle(X)
#np.random.shuffle(y)
#x_train,y_train = X[:800],y[:800]
#x_test,y_test = X[800:],y[800:]
#plt.scatter(x_train,y_train)
#plt.show()
#print x_test,y_test
#plt.scatter(x_train,y_train)
#plt.show()
def my_function(x_train,y_train):

    model = Sequential()
    model.add(Dense(1000,input_dim = 1))
    #model.add(Dense(20,input_shape(1,)))
    #
    model.add(Activation('relu'))
    #model.add(Dense(50))
    #model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(lr = 0.001)
    sgd = SGD(lr = 0.1,decay=12-5,momentum=0.9)
    model.compile(optimizer = adam,loss = 'mse')
    print '-------------training--------------'
    model.fit(x_train,y_train,batch_size= 12,nb_epoch = 500,shuffle=True)
#    for step in range(6000):
#        cost = model.train_on_batch(x_train,y_train)
#        if step %10 == 0:
#            try:
#                ax.lines.remove(lines[0])
#            except Exception:
#                pass
#            Y_train_pred = model.predict(x_train)
#            lines = ax.plot(x_train,Y_train_pred,'r-',lw =5)
#            plt.pause(0.01)
            #print 'train cost:',cost'''
    #print '--------------testing----------------'
    #cost = model.evaluate(x_test,y_test,batch_size= 5)
    #print 'test_cost\n',cost 
    #Y_pred = model.predict(x_test)
    Y_train_pred=model.predict(x_train)
    #plt.scatter(x_test,y_test)
    #plt.plot(x_test,Y_pred)
    plt.scatter(x_train,y_train)
    plt.plot(x_train,Y_train_pred,'r-',lw=5)
    plt.show()
#fig = plt.figure()
#ax =fig.add_subplot(1,1,1)
#ax.scatter(x_train,y_train)
#plt.ion()
#plt.show()
my_function(x_train,y_train)
