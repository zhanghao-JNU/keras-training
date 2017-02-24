import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import Adam
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 50
epsilon_std = 1.0

x = Input(batch_shape = (batch_size,original_dim))
h = Dense(intermediate_dim,activation= 'relu')(x)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean,z_log_var = args
    epsilon = K.random_normal(shape = (batch_size,latent_dim),mean = 0.,
            std = epsilon_std)
    return z_mean + K.exp(z_log_var/2)*epsilon
z = Lambda(sampling,output_shape = (latent_dim,))([z_mean,z_log_var])

decoder_h = Dense(intermediate_dim,activation = 'relu')
decoder_mean = Dense(original_dim,activation = 'sigmoid')

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x,x_decoded_mean):
    xent_loss = original_dim*objectives.binary_crossentropy(x,x_decoded_mean)
    kl_loss = -0.5 * K.sum(1+z_log_var - K.square(z_mean)-K.exp(z_log_var),axis = -1)
    return xent_loss +kl_loss

vae = Model(x,x_decoded_mean)
vae.compile(optimizer = Adam(),loss = vae_loss)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

vae.fit(x_train,x_train,shuffle = True,nb_epoch = nb_epoch,batch_size = batch_size,validation_data = (x_test,x_test))
encoder = Model(x,z_mean)
x_test_encoded = encoder.predict(x_test,batch_size = batch_size)
print x_test_encoded.shape

plt.figure(figsize = (6,6))
plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c = y_test)
plt.colorbar()

decoder_input = Input(shape = (latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input,_x_decoded_mean)

n = 15
digit_size = 28
figure = np.zeros((digit_size*n,digit_size*n))

grid_x = np.linspace(-15,15,n)
grid_y = np.linspace(-15,15,n)

for i,yi in enumerate(grid_x):
    for j,xi in enumerate(grid_y):
        z_sample = np.array([[xi,yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size,digit_size)
        figure[i*digit_size:(i+1)*digit_size,
                j*digit_size:(j+1)*digit_size] = digit

plt.figure(figsize = (10,10))
plt.imshow(figure)
plt.show()
