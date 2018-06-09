# -*- coding:utf-8 -*-
'''
function:implement autoencoder
'''
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import  Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()
# Data processing
x_train=x_train.astype('float32')/255.0-0.5  # minmax normalized
x_test=x_test.astype('float32')/255.0-0.5    # minmax normalized
x_train=x_train.reshape((x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))
print(x_train.shpae)
print(x_test.shape)

encoding_dim=2
input_img=Input(shape=(784,))
# Encoder layer
encoded=Dense(128,activation='relu')(input_img)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoded_output=Dense(encoding_dim)(encoded)

# Decoder layer
decoded=Dense(10,activation='relu')(encoded_output)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded=Dense(784,activation='relu')(decoded)

# Construct the autoencoder model
autoencoder=Model(input=input_img,output=decoded)
# construct the encoder model for plotting
encoder=Model(optimizer='adam',loss='mse')
# compile autoencoder
autoencoder.compile(optimizer='adam',loss='mse')

# Train
autoencoder.fit(x_train,y_train,epochs=20,batch_size=256,shuffle=True)

# Plotting
encoded_imgs=encoder.predict(x_test)
plt.scatter(encoded_imgs[:,0],encoding_dim[:,1],c=y_test)
plt.colorbar()
plt.show()