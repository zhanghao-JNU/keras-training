from keras.datasets import mnist
import numpy as np
from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,UpSampling2D
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt

np.random.seed(1337)

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))
print x_train.shape
print x_test.shape

noise_factor = 0.5
x_train_noisy = x_train + noise_factor*np.random.normal(loc=0.0,scale = 1.0,size = x_train.shape)
x_test_noisy = x_test + noise_factor*np.random.normal(loc = 0.0,scale = 1.0,size = x_test.shape)
x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_test_noisy = np.clip(x_test_noisy,0.,1.)

input_img = Input(shape = (28,28,1))

x = Convolution2D(32,3,3,activation = 'relu',border_mode = 'same')(input_img)
x = MaxPooling2D((2,2),border_mode = 'same')(x)
x = Convolution2D(32,3,3,activation = 'relu',border_mode = 'same')(x)
encoded = MaxPooling2D((2,2),border_mode = 'same')(x)

x = Convolution2D(32,3,3,activation = 'relu',border_mode = 'same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(32,3,3,activation = 'relu',border_mode = 'same')(x)
x = UpSampling2D((2,2))(x)
decoded = Convolution2D(1,3,3,activation = 'sigmoid',border_mode = 'same')(x)

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer = Adam(),loss = 'binary_crossentropy')
autoencoder.fit(x_train_noisy,x_train,nb_epoch = 2,batch_size = 128,shuffle = True,validation_data = (x_test_noisy,x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize = (20,4))
for i in range(10):
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3,n,i+n+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3,n,i+(2*n)+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

