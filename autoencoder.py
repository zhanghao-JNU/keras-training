from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

(x_train,_),(x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape
'''
#simple auto_encoder
encoding_dim = 32
input_img = Input(shape = (784,))

encoded = Dense(encoding_dim,activation = 'relu')(input_img)
decoded = Dense(784,activation = 'sigmoid')(encoded)

autoencoder = Model(input = input_img,output = decoded)

encoder = Model(input = input_img,output= encoded)

encoded_input = Input(shape = (encoding_dim,))
decoder_layer = autoencoder.layers[-1]
print autoencoder.layers[0]
decoder = Model(input = encoded_input,output = decoder_layer(encoded_input))

autoencoder.compile(optimizer = Adam(),loss = 'binary_crossentropy')



autoencoder.fit(x_train,x_train,nb_epoch = 20,batch_size = 256,shuffle = True,validation_data = (x_test,x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

'''
#add a Dense layer with a l1 activity regularizer
from keras import regularizers
encoding_dim = 32
input_img = Input(shape = (784,))
encoded = Dense(encoding_dim,activation = 'relu',activity_regularizer = regularizers.activity_l1(10e-5))(input_img)
decoded = Dense(784,activation = 'sigmoid')(encoded)
autoencoder = Model(input = input_img,output = decoded)
autoencoder.compile(optimizer = Adam(),loss = 'binary_crossentropy')
autoencoder.fit(x_train,x_train,nb_epoch = 50,batch_size = 256,shuffle = True,validation_data= (x_test,x_test))'''
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


