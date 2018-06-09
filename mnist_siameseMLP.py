# -*- coding:utf-8 -*-
''''
Train a siamese MLP on pairs of digits from the mnist dataset
'''
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Flatten,Dense,Dropout,Lambda
from keras.optimizers import  RMSprop
from keras import backend as k

num_classes=10
epochs=20

# 计算两个向量之间的欧式距离
def euclidean_distance(vects):
    x,y=vects
    return k.sqrt(k.maximum(k.sum(k.square(x-y),axis=1,keepdims=True),k.epsilon()))
def eucl_dist_output_shape(shapes):
    shape1,shape2=shapes
    return(shape1[0],1)

def contrastive_loss(y_true,y_pred):
    margin=1
    return k.mean(y_true*k.square(y_pred)+(1-y_true)*k.square(k.maximum(margin-y_pred,0)))

def create_pairs(x,digit_indices):
    '''
    positive and negative pairs creation
    :param x:label
    :param digit_indices:
    :return:
    '''
    pairs=[]
    labels=[]
    n=min([len(digit_indices[d]) for d in range(num_classes)])-1
    for d in range(num_classes):
        for i in range(n):
            z1,z2=digit_indices[d][i],digit_indices[d][i+1]
            pairs+=[[x[z1],x[z2]]]
            inc=random.randrange(1,num_classes)
            dn=(d+inc)%num_classes
            z1,z2=digit_indices[d][i],digit_indices[d][i]
            pairs+=[x[z1],x[z2]]
            labels+=[1,0]

    return np.array(pairs),np.array(labels)

def create_base_network(input_shape):
    '''
    Base network to be shared.
    :param input_shape:
    :return:
    '''
    input=Input(shape=input_shape)
    x=Flatten()
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Dense(128,activation='relu')(x)
    return Model(input,x)

def compute_accuracy(y_true,y_pred):
    '''
    Compute classification accuracy with a fixed thershold on distance
    :param y_true:
    :param y_pred:
    :return:
    '''
    pred=y_pred.ravel()<0.5
    return np.mean(pred==y_pred)

def accuracy(y_true,y_pred):
    '''
    compute classification accuracy wiht a fixed threshold on distance
    :param y_true:
    :param y_pred:
    :return:
    '''
    return k.mean(k.equal(y_true,k.cast(y_pred<0.5,y_true.dtype)))

# the data ,split between train and test sets
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
input_shape=x_train.shape[1:]

# create training data postive and negative pairs
digit_indices=[np.where(y_train==i)[0] for i in range(num_classes)]
tr_pairs,tr_y=create_pairs(x_train,digit_indices)

digit_indices=[np.where(y_test==i)[0] for i in range(num_classes)]
te_pairs,te_y=create_pairs(x_test,digit_indices)

# network defination
base_network=create_base_network(input_shape)

input_a=Input(shape=input_shape)
input_b=Input(shape=input_shape)

#bacause we reuse the same instance 'base_network'
# the weights of the network will be shared across the two branches
process_a=base_network(input_a)
process_b=base_network(input_b)
distance=Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([process_a,process_b])

model=Model([input_a,input_b],distance)

# Train
rms=RMSprop()
model.compile(loss=contrastive_loss,optimizer=rms,metrics=[accuracy])
model.fit([tr_pairs[:,0]],tr_pairs[:,1],tr_y,batch_size=128,epochs=epochs,validation_data=(te_pairs[:,0],te_pairs[:,1],te_y))

#compute final accuracy on training and test data set
y_pred=model.predict([tr_pairs[:,0],tr_pairs[:,1]])
tr_acc=compute_accuracy(tr_y,y_pred)
y_pred=model.predict([te_pairs[:,0]],te_pairs[:,1])
te_acc=compute_accuracy(te_y,y_pred)

print(' Accuracy on training set:%0.2f%%'%tr_acc*100)
print(' Accuracy on testing set:%0.2f%%'%te_acc*100)
