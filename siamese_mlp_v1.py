# -*- coding:utf-8 -*-
'''
随机生成一批数据
创建多标签分类模型
并评估模型在不同标签下的准确率
'''
from keras.layers import Dense,Dropout,Flatten,Input,Lambda
import  keras.backend as K
import numpy as np
import random
from keras.optimizers import RMSprop
from keras.models import Model
import keras
# 用于选择每个标签的阈值
from sklearn.metrics import matthews_corrcoef,hamming_loss

def generate_random_data():
    pairs=[]
    labels=[]
    x1=np.random.uniform(0,1,size=(10,5,3))
    x2=np.random.uniform(1,2,size=(10,5,3))
    # print('x1:',x1)
    # print('x2:',x2)
    for i in range(len(x1)-1):
        pairs+=[[x1[i],x1[i+1]]]
        labels+=[[1,0]]
    for i in range(len(x2)-1):
        pairs+=[[x2[i],x2[i+1]]]
        labels+=[[1,0]]
        pairs+=[[x1[i],x2[i+1]]]
        labels+=[[0,1]]
    return np.array(pairs),np.array(labels)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    print(y_true.shape,y_pred.shape)
    for i in range(len(y_pred.shape[2])):
        K.mean(K.equal())
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
def euclidean_distance(vects):
    x, y = vects
    ##K.epsilon()以数值形式返回一个（一般来说很小的）数，用以防止除0错误
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print('shape1:',shape1)
    print('shape2:',shape2)
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    print('y_true:',y_true)
    print('y_pred:',y_pred)
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def totalAccuracy(y_true,y_pred):
    pass


if __name__ == '__main__':
    # Generate some random data
    tr_pairs,tr_y=generate_random_data()
    te_pairs,te_y=generate_random_data()
    print(tr_pairs.shape)
    print(tr_y.shape)

    input_shape=(5,3)
    epochs=100
    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)


    x=keras.layers.concatenate([processed_a,processed_b])
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.1)(x)
    x=Dense(2,activation='sigmoid')(x)

    model = Model([input_a, input_b], x)

    # train
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    print('tr_y:',tr_y)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
      batch_size=128,
      epochs=epochs,
      validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    out = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    print(type(out))

    # 多标签分类要为每个标签设定阈值  以下是阈值的选择方法
    threshold=np.arange(0.1,0.9,0.1)
    accuracies=[]
    best_threshold=np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        acc=[]
        y_prob=np.array(out[:,i])
        for t in threshold:
            y_pred=[1 if prob>=t else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(tr_y[:,i],y_pred))
        acc=np.array(acc)
        print('acc:', acc)
        index=np.argmax(acc)
        best_threshold[i]=threshold[index]
        accuracies.append(np.max(acc))

    print('best_threshold:',best_threshold)
    print('The Accuracy of Each Label:',accuracies)
    y_pred=np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(te_y.shape[1])] for i in range(len(te_y))])
    print('y_pred:',y_pred)
    print('te_y:',te_y)
    hamming_loss_value=hamming_loss(tr_y,y_pred)
    print('The hamming loss between them is：',hamming_loss_value)
    total_correctly_predicted = len([i for i in range(len(te_y)) if (te_y[i] == y_pred[i]).sum() == 2])
    print('Total_correctly_predicted number:',total_correctly_predicted)
    print('The rate of total correctly predicted number is:',total_correctly_predicted*1.0/len(te_y))



