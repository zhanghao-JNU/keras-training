from __future__ import print_function
from six.moves import cPickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Bidirectional
from keras.datasets import imdb

max_features = 20000
maxlen = 100
batch_size = 32
print ("loading datai............")
(x_train,y_train),(x_test,y_test) = imdb.load_data(nb_words = max_features)
print (x_trian[0])
