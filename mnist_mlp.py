# -*- coding:utf-8 -*-
'''
Train a simple deep NN on the mnist dataset

'''
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop

# Parameters
batch_size=128
num_classes=10
epoches=5

# the data, split between train and test set
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# reshape the data
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train/=255
x_test/=255
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
print('train sample shape:',x_train.shape)

# Convert class vectors to binary class matrics
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# build the model
model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu',name='layer_name'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
# present the network
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epoches,verbose=1,validation_data=(x_test,y_test))
score=model.evaluate(x_test,y_test,verbose=0)
print('Train LOSS:',score[0])
print('TEST LOSS:',score[1])