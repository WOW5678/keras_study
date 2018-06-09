# -*- coding:utf-8 -*-
'''

'''
from __future__ import print_function
from keras.layers import Dense,Input
from keras import layers
from keras.datasets import mnist
import keras.backend as K
import keras
from keras.models import Sequential
class Antirectofier(layers.Layer):
    def compute_output_shape(self, input_shape):
        shape=list(input_shape)
        assert len(shape)==2 # only valid for 2D tensor
        shape[-1]*=2
        return tuple(shape)
    def call(self,inputs):
        # 求出每一列的平均值
        inputs-=K.mean(inputs,axis=1,keepdim=True)
        inputs=K.l2_normalize(inputs,axis=1)
        pos=K.relu(inputs)
        neg=K.relu(-inputs)
        return K.concatenate([pos,neg],axis=1)

# Global parameters
batch_size=128
num_classes=10
epochs=40

# split the data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)

# Convert class to binary class vectors
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# Build the model
model=Sequential()
model.add(layers.Dense(256,input_shape=(784,)))
model.add(Antirectofier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectofier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# Train the model
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
