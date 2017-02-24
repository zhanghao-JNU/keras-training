import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.optimizers import Adam
nb_classes = 10
img_rows,img_cols = 28,28

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print x_train.shape
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
print x_train.shape
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

def make_model(dense_layer_sizes,nb_filters,nb_conv,nb_pool):
    model = Sequential()
    model.add(Convolution2D(
        nb_filters,
        nb_conv,
        nb_conv,
        border_mode = 'valid',
        input_shape = (img_rows,img_cols,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters,
        nb_conv,
        nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(),
            metrics = ['accuracy'])
    return model
dense_size_candidates = [[32],[64],[32,32],[64,64]]
my_classifier = KerasClassifier(make_model,batch_size = 32)
validator = GridSearchCV(my_classifier,
        param_grid = {
            'dense_layer_sizes':dense_size_candidates,
            'nb_epoch':[3,6],
            'nb_filters':[8],
            'nb_conv':[3],
            'nb_pool':[2]
            },
        scoring = 'log_loss',
        n_jobs =1)
validator.fit(x_train,y_train)

print 'the patameters of the best model are:',validator.best_params_

best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test,y_test)
print '\n'
for metric,value in zip(metric_names,metric_values):
    print metric,':',value
