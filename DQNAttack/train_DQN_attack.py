# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/11 0011 下午 8:12
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用DQN学习一个攻击网络
 即学习一种像素更改策略，通过更改极少的像素来改变模型的预测结果
"""
import numpy as np
import keras
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import combinations,product
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
from helper_fun import *
import random

if __name__ == '__main__':
    dataset='MNIST'
    #dataset='CIFAR'

    if dataset=='MNIST':
        input_model = load_model('data/mnist_classification_model')
        mnist=input_data.read_data_sets('MNIST_DATA',one_hot=True)
        mnist_images=mnist.train.images
        mnist_train_images=np.reshape(mnist_images,[mnist_images.shape[0],28,28,1])
        X_train=mnist_train_images
        y_train=mnist.train.labels

        mnist_images=mnist.test.images
        X_test=np.reshape(mnist_images,[mnist_images.shape[0],28,28,1])
        y_test=mnist.test.labels
        input_shape=[28,28,1]

        num_classes=10
        lamda=0.4

    if dataset=='CIFAR':
        input_model=load_model('data/CIFAR_classification_model')
        cifar_class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        (X_train,y_train),(X_test,y_test)=cifar10.load_data()
        num_classes=10
        y_train=keras.utils.to_categorical(y_train,num_classes)
        y_test=keras.utils.to_categorical(y_test,num_classes)

        X_train=X_train/255.0
        X_test=X_test/255.0
        input_shape=[32,32,3]
        lamda=1

    #define action space of agent
    block_size=2
    x_span=list(range(0,X_train.shape[1],block_size))
    blocks=list(product(x_span,x_span)) #list:[(0,0),(0,2)....(2,0),(2,2)...]
                                        #len (blocks):14*14=196


    #define DQN model
    cnn_model=get_cnn_model(input_shape=input_shape)
    dnn_model=get_dnn_model(input_shape=[num_classes])

    img_input=Input(shape=input_shape)
    prob_input=Input(shape=[num_classes])

    prob_rep=dnn_model(prob_input) #得到预测向量的表示（batchsize,128）
    img_rep=cnn_model(img_input) #得到图片的表示  (batchsize,128)


    x=Concatenate(axis=-1)([prob_rep,img_rep]) #列数增加了 x:(batchsize,128*2)
    x=Dense(len(blocks),activation='linear')(x) #最终的Q结果值（batchsize,196）

    q_model=Model(inputs=[img_input,prob_input],outputs=[x])
    q_model.summary()
    q_model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

    #Q learning parameters
    eps=0.9 #探索率
    dis_factor=0.9 #奖励折扣因子
    max_memory=1000 #经验池空间
    max_blocks_attack=15 #每个episode中最大的尝试次数

    success=[]
    success_rate=[]
    exp_replay=[] #保存着每一次的尝试经验

    ### generating espisodes from 5k test images
    for games in range(5000):
        sample_img=X_test[games] #(28,28,1)
        sample_img=np.expand_dims(sample_img,axis=0) #(1,28,28,1)

        if games>0 and games%300==0:
            eps=eps-0.1
        if eps<=0.1:
            eps=0.1

        orig_label=np.argmax(input_model.predict(sample_img),axis=1) #axis=1取每行中的最大值的索引 如:[7]
        orig_img=np.array(sample_img)

        for ite in range(0,max_blocks_attack):
            sample_img_prob=input_model.predict(sample_img)[0] #预测概率值，列表[,,,,,,]
            sample_img_prob=np.expand_dims(sample_img_prob,axis=0) #(?,num_classese) 满足q_model的输入要求

            if np.random.rand() < eps:
                action = np.random.randint(0, len(blocks)) #随机选择action
            else:
                action=np.argmax(q_model.predict([sample_img,sample_img_prob])) #模型预测

            attack_region=np.zeros((sample_img.shape))#（1,28,28,1）
            attack_cord=blocks[action] #得到一个元素如（12,2）表示的是区域的坐标
            #将attack_cord中坐标指明的区域像素都设置为1
            attack_region[0, attack_cord[0]:attack_cord[0] + block_size, attack_cord[1]:attack_cord[1] + block_size,:] = 1

            sample_img_noise=sample_img+lamda*attack_region
            sample_img_noise_prob = input_model.predict(sample_img_noise)
            mod_label = np.argmax(input_model.predict(sample_img_noise), axis=1)

            if mod_label!=orig_label:
                reward=10
                success.append(1)
                exp_replay.append([sample_img,sample_img_prob,action,reward,sample_img_noise,sample_img_noise_prob])
                break
            else:
                reward=-0.1
                exp_replay.append([sample_img, sample_img_prob, action, reward, sample_img_noise, sample_img_noise_prob])

            sample_img=np.array(sample_img_noise) #更改状态

        if ite==(max_blocks_attack-1):
            reward=-1
            exp_replay.append([sample_img,sample_img_prob,action,reward,sample_img_noise,sample_img_noise_prob])
            success.append(0)

        if len(exp_replay)>=max_memory:
            q_model=update_q_model(exp_replay=exp_replay,q_model=q_model, batch_size = 32, dis_factor = dis_factor)
            exp_replay=[]

            print('Q model update','success rate',np.mean(np.array(success)))
            success_rate.append(np.mean(np.array(success)))
            success=[]


    # saving q model and success rate
    q_model.save(dataset+'_Q_adversial_model')





