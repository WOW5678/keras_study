# -*- coding:utf-8 -*-
'''
function: show how to build a variational autoencoder with kears
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input,Lambda,Dense
from keras.models import  Model
from keras import backend as k
from keras import metrics
from keras.datasets import mnist

batch_size=100
original_dim=784
latent_dim=2
intermediate_dim=256
epochs=50
epsilon_std=1.0

x=Input(shape=(original_dim,))
h=Dense(intermediate_dim,activation='relu')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)

def sampling(args):
    z_mean,z_log_var=args
    epsilon=k.random_normal(shape=(k.shape(z_mean)[0],latent_dim),mean=0.0,stddev=epsilon_std)
    return z_mean+k.exp(z_log_var/2)*epsilon

z=Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_var])

# We instentiate these layers separately so as to reuse them later
decoder_h=Dense(intermediate_dim,activation='relu')
decoder_mean=Dense(original_dim,activation='sigmoid')
h_decoded=decoder_h(z)
#x_decoded_mean为最后还原的x
x_decoded_mean=decoder_mean(h_decoded)

# Instantiate VAE model
vae=Model(x,x_decoded_mean)

# Compute VAE loss
xent_loss=original_dim*metrics.binary_crossentropy(x,x_decoded_mean)
kl_loss=-0.5*k.sum(1+z_log_var-k.square(z_mean)-k.exp(z_log_var),axis=-1)
vae_loss=k.mean(xent_loss+kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop',loss=None)
vae.summary()

# Train the VAE on Mnist digits
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
x_train=x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
# np.prod()所有元素相乘的结果
x_test=x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test,None)
)
# build a model to project inputs on the latent space
encoder=Model(x,z_mean)
x_test_encoded=encoder.predict(x_test,batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c=y_test)
plt.colorbar()
plt.show()

# Build a digit generator that can sample from  the learned distribution
decoder_input=Input(shape=(latent_dim,))
_h_decoded=decoder_h(decoder_input)
_x_decoded_mean=decoder_mean(_h_decoded)
generator=Model(decoder_input,_x_decoded_mean)

# Display a 2D manifold of the digits
n=15
digit_size=28
figure=np.zeros((digit_size*n,digit_size*n))
#求概率密度函数指定点的函数值
#stats.norm.pdf正态分布概率密度函数
# grid_x 与grid_y完全相同
grid_x=norm.ppf(np.linspace(0.05,0.95,n))
grid_y=norm.ppf(np.linspace(0.05,0.95,n))

for i,yi in enumerate(grid_x):
    for j,xi in enumerate(grid_y):
        z_sample=np.array([[xi,yi]])
        x_decoded=generator.predict(z_sample)
        digit=x_decoded[0].reshape(digit_size,digit_size)
        figure[i*digit_size:(i+1)*digit_size,
        j*digit_size:(j+1)*digit_size]=digit
plt.figure(figsize=(10,10))
plt.imshow(figure,cmap='Greys_r')
plt.show()
