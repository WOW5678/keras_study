# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/13 0013 下午 4:34
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: batch Normalization
"""
from keras.layers.normalization import  BatchNormalization
from keras.models import  Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import  SGD

# 样本模型
model=Sequential()
# 我们可以把这一大块作为输入层
model.add(Dense(64,input_dim=14,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

#可以把这一块作为隐藏层
model.add(Dense(64,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

#我们可以将这一块作为一个输出层
model.add(Dense(2,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#setting up the optimization of our weights
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd)

#running the fitting
#fit中的verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
# 注意： 默认为 1

# evaluate 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# 注意： 只能取 0 和 1；默认为 1
model.fit(x_train,y_train,nb_epoch=20,batch_size=16,show_accuracy=True,validation_split=0.2, verbose = 2)