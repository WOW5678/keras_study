# -*- coding:utf-8 -*-
'''
This script demonstrates how to build a variational autoencoder
with keras and deconvolution layers
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input,Dense,Lambda,Flatten,Reshape
from keras.layers import Conv2D,Conv2DTranspose
from keras.models import  Model
from keras import backend as k
from keras import metrics
from keras.datasets import mnist

# Input image dimensions
img_rows,img_cols,img_chns=28,28,1
# number of convolutional filters to use
filters=64
# convolution kernel size
num_conv=3

batch_size=100
if k.image_data_format()=='channels_first':
    original_img_size=(img_chns,img_rows,img_cols)
else:
    original_img_size=(img_rows,img_cols,img_chns)
latent_dim=2
intermediate_dim=128
epsilon_std=1.0
epochs=5

x=Input(shape=original_img_size)
conv_1=Conv2D(img_chns,kernel_size=(2,2),padding='same',activation='relu')(x)
conv_2=Conv2D(filters,kernel_size=(2,2),padding='same',activation='relu',strides=(2,2))(conv_1)
conv_3=Conv2D(filters,kernel_size=num_conv,padding='same',activation='relu',strides=1)(conv_2)
conv_4=Conv2D(filters,kernel_size=num_conv,padding='same',activation='relu',strides=1)(conv_3)
flat=Flatten()(conv_4)
hidden=Dense(intermediate_dim,activation='relu')(flat)

z_mean=Dense(latent_dim)(hidden)
z_log_var=Dense(latent_dim)(hidden)

def sampling(args):
    z_mean,z_log_var=args
    epsilon=k.random_normal(shape=(k.shape(z_mean)[0],latent_dim),mean=0.,stddev=epsilon_std)
    return z_mean+k.exp(z_log_var)*epsilon

# note that "output_shpae" isn't necessary with the tensorflow backend
# so you could write "lambda(sampling)([z_mean,z_log_var])"
z=Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_var])
# we instantiate these layers seprately so as to reuse them later
decoder_hid=Dense(intermediate_dim,activation='relu')
decoder_upsample=Dense(filters*14*14,activation='relu')

if k.image_data_format()=='channels_first':
    output_shape=(batch_size,filters,14,14)
else:
    output_shape=(batch_size,14,14,filters)

decoder_reshape=Reshape(output_shape[1:])
decoder_deconv_1=Conv2DTranspose(filters,kernel_size=num_conv,
                                 padding='same',
                                 strides=1,
                                 activation='relu')
decoder_deconv_2=Conv2DTranspose(filters,kernel_size=num_conv,
                                 padding='same',
                                 activation='relu',
                                 strides=1)
if k.image_data_format()=='channels_first':
    output_shape=(batch_size,filters,28,28)
else:
    output_shape=(batch_size,28,28,filters)

decoder_deconv_3_upsamp=Conv2DTranspose(filters,
                                        kernel_size=(3,3),
                                        strides=(2,2),
                                        padding='valid',
                                        activation='relu')
decoder_mean_squash=Conv2D(img_chns,kernel_size=2,padding='valid',activation='sigmoid')

hid_decoded=decoder_hid(z)
up_decoded=decoder_upsample(hid_decoded)
reshape_decoded=decoder_reshape(up_decoded)
deconv_1_decoded=decoder_deconv_1(reshape_decoded)
deconv_2_decoded=decoder_deconv_2(deconv_1_decoded)
x_decoded_relu=decoder_deconv_3_upsamp(deconv_2_decoded)
# ????
x_decoded_mean_squash=decoder_mean_squash(x_decoded_relu)

# instantiate VAE model
vae=Model(x,x_decoded_mean_squash)

# Compute VAE loss
xent_loss=img_rows*img_cols*metrics.binary_crossentropy(
    k.flatten(x),
    k.flatten(x_decoded_mean_squash)
)
kl_loss=-0.5*k.sum(1+z_log_var-k.square(z_mean)-k.exp(z_log_var),axis=-1)
vae_loss=k.mean(xent_loss+kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop',loss=None)
vae.summary()

# Train the VAE on Mnist digits
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype('float32')/255
x_train=x_train.reshape((x_train.shape[0],)+original_img_size)
x_test=x_test.astype('float32')/255
x_test=x_test.reshape((x_test.shape[0],)+original_img_size)

print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)

vae.fit(x_train,shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test,None))

# build a model to project inputs on th latent space
encoder=Model(x,z_mean)

x_test_encoded=encoder.predict(x_test,batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c=y_test)
plt.colorbar()
plt.show()

# build a digit generatoor that can sample from the learned distribution
decoder_input=Input(shape=(latent_dim,))
_hid_decoded=decoder_hid(decoder_input)
_up_decoded=decoder_upsample(_hid_decoded)
reshape_decoded=decoder_reshape(_up_decoded)
_deconv_1_decoded=decoder_deconv_1(reshape_decoded)
_deconv_2_decoded=decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu=decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash=decoder_mean_squash(_x_decoded_relu)
generator=Model(decoder_input,_x_decoded_mean_squash)

# Display a 2D manifold of the digits
n=15
digit_size=28
figure=np.zeros((digit_size*n,digit_size*n))
grid_x=norm.ppf(np.linspace(0.05,0.95,n))
grid_y=norm.ppf(np.linspace(0.05,0.95,n))

for i,xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample=np.array([[xi,yi]])
        z_sample=np.tile(z_sample,batch_size).reshape(batch_size,2)
        x_decoded=generator.predict(z_sample,batch_size=batch_size)
        digit=x_decoded[0].reshape(digit_size,digit_size)
        figure[i*digit_size:(i+1)*digit_size,j*digit_size:(j+1)*digit_size]=digit_size
plt.figure(figsize=(10,10))
plt.imshow(figure,cmap='Greys_r')
plt.show()