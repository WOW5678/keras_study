# -*- coding:utf-8 -*-
'''
Train a recurrent convolutional network on the IMDB sentiment classificaiton task
cnn_lstm 首先对数据进行卷积操作，然后将卷积之后的结果送入LSTM，接上全连接层进行分类
'''
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM,Conv1D,MaxPooling1D
from keras.datasets import imdb

# Embedding
max_features=20000
maxlen=100
embedding_size=128

# Convoluation
kernel_size=5
filter=64
pool_size=4

# LSTM
lstm_output_size=70
# Training
batch_size=30
epochs=2

print('Loading the dadaset...')
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print(len(x_train),'train sequences')
print(len(x_test),'test sequences')

print('Pading sequences(samples x time)')
x_train=sequence.pad_sequences(x_train,maxlen)
x_test=sequence.pad_sequences(x_test,maxlen)
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)


print('Building the model...')
model=Sequential()
model.add(Embedding(max_features,embedding_size,input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filter,kernel_size=kernel_size,padding='valid',activation='relu',strides=1))
print('model.output_shape:',model.output_shape) # (None, 96, 64)
model.add(MaxPooling1D(pool_size=pool_size))
print('model.output_shape:',model.output_shape) # (None, 24, 64)
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
# 因为全全连接层只有1个神经元，所以使用的激活函数为softmax
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train')
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))

score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)
print('Test Score:',score)
print('Test Accuracy:',acc)
