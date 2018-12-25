# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/25 0025 上午 10:52
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用一个n-gram多通道卷积神经网络进行情绪分析
"""
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump
from pickle import load

from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import  pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Dense,Input,Dropout,Embedding,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import load_model

#将文档加载到内存中
def load_doc(filename):
    #只读的形式打开文件
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

#将文档转变成clean tokens
def clean_doc(doc):
    #使用空格将句子分割成tokens
    tokens=doc.split()
    #移除标点符号的token
    table=str.maketrans('','',punctuation)
    tokens=[w.translate(table) for w in tokens]
    #移除不是字符的token
    tokens=[word for word in tokens if word.isalpha()]
    #过滤掉stopwords
    stop_words=set(stopwords.words('english'))
    tokens=[w for w in tokens if not w in stop_words]
    #将单个的token拼接成一句话
    tokens=' '.join(tokens)
    return tokens
#加载一个目录下的所有的doc文档
def process_docs(directory,is_train):
    documents=list()
    #遍历文件夹中所有的文件
    for filename in listdir(directory):
        #在训练模式下 移除在测试集中的reviews(100条数据做测试)
        #这里选择最后的1000条做测试 即开头为cv9
        if is_train and filename.startswith('cv9'):
            continue
        #在测试模型下 移除训练集中的reviews(前900条数据)
        if not is_train and not filename.startswith('cv9'):
            continue
        #创建full path 打开文件
        path=directory+'/'+filename
        #load the doc
        doc=load_doc(path)
        #clearn doc
        #tokens是一个句子（清理完之后的）
        tokens=clean_doc(doc)
        #add to list
        documents.append(tokens)
    return documents

#保存数据集到文件中
def save_dataset(dataset,filename):
    dump(dataset,open(filename,'wb'))
    print('saved:%s'%filename)

#加载clean的数据集
def load_dataset(filename):
    return load(open(filename,'rb'))

#训练tokens
def create_tokenizer(lines):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
#计算最大的文档长度
def max_length(lines):
    return max([len(s.split()) for s in lines])
#编码a list of lines
def encode_text(tokenizer,lines,length):
    #将文本转换成integer
    encoded=tokenizer.texts_to_sequences(lines)
    #pad encoded sequences
    padded=pad_sequences(encoded,maxlen=length,padding='post')
    return padded

#定义模型
def define_model(length,vocab_size):
    #chanel 1
    input1=Input(shape=(length,))
    embedding1=Embedding(vocab_size,100)(input1)
    conv1=Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1=Dropout(0.5)(conv1)
    pool1=MaxPooling1D(pool_size=2)(drop1)
    flat1=Flatten()(pool1)

    #chanel2
    input2=Input(shape=(length,))
    embedding2=Embedding(vocab_size,100)(input2)
    conv2=Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
    drop2=Dropout(0.5)(conv2)
    pool2=MaxPooling1D(pool_size=2)(drop2)
    flat2=Flatten()(pool2)

    #chanel3
    input3=Input(shape=(length,))
    embedding3=Embedding(vocab_size,100)(input3)
    conv3=Conv1D(filters=32,kernel_size=8,activation='relu')(embedding3)
    drop3=Dropout(0.5)(conv3)
    pool3=MaxPooling1D(pool_size=2)(drop3)
    flat3=Flatten()(pool3)

    #merge
    merged=concatenate([flat1,flat2,flat3])
    #interpretation
    dense1=Dense(10,activation='relu')(merged)
    outputs=Dense(1,activation='sigmoid')(dense1)
    model=Model(inputs=[input1,input2,input3],outputs=outputs)
    #compile
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    #summarize
    print(model.summary())
    #plot_model(model,show_shapes=True,to_file='multichannel.png')
    return model


if __name__ == '__main__':
    #加载所有的训练的reviews
    negative_docs=process_docs('review_polarity\\txt_sentoken\\neg',True)
    positive_docs=process_docs('review_polarity\\txt_sentoken\pos',True)
    trainX=negative_docs+positive_docs
    trainY=[0 for _ in range(900)]+[1 for _ in range(900)]
    save_dataset([trainX,trainY],'review_polarity\\train.pkl')

    #加载所有的测试的reviews
    negative_docs=process_docs('review_polarity\\txt_sentoken\\neg',False)
    positive_docs=process_docs('review_polarity\\txt_sentoken\\pos',False)
    trainX=negative_docs+positive_docs
    trainY=[0 for _ in range(100)]+[1 for _ in range(100)]
    save_dataset([trainX,trainY],'review_polarity\\test.pkl')

    #设计多通道卷积网络模型
    #加载训练样本
    trainLines,trainLabels=load_dataset('review_polarity\\train.pkl')
    print(trainLines[:2])
    testLines, testLabels = load_dataset('review_polarity\\test.pkl')
    print(testLines[:2])
    #create tokens
    tokenizer=create_tokenizer(trainLines)
    #计算最大的文档长度
    length=max_length(trainLines)
    #计算文档长度
    vocab_size=len(tokenizer.word_index)+1
    print('Max document length:%d'%length) #1382
    print('Vocab size:%d'%vocab_size) #44295

    #编码数据
    trainX=encode_text(tokenizer,trainLines,length)
    testX=encode_text(tokenizer,testLines,length)
    print(trainX.shape,testX.shape)  # (1800, 1382)
    #定义模型
    model=define_model(length,vocab_size)
    #fit model
    model.fit([trainX,trainX,trainX],np.array(trainLabels),epochs=1,batch_size=16)
    #save the model
    model.save('review_polarity\model.h5')

    #评估模型
    #加载模型
    model=load_model('review_polarity\model.h5')
    #评估模型在训练集上的表现
    loss,acc=model.evaluate([trainX,trainX,trainX],np.array(trainLabels),verbose=0)
    print('Train Accuracy:%.3f'%(acc*100))

    #评估模型在测试集上的表现
    loss,acc=model.evaluate([testX,testX,testX],np.array(testLabels),verbose=0)
    print('Test Accuracy:%.3f'%(acc*100))




