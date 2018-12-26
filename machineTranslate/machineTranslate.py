# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/26 0026 上午 9:45
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 从头开发一个神经翻译系统
 需要掌握的知识：
 1.数据准备工作，数据清理
 2.encoder-decoder model
 3.评估训练好的模型 并用于新数据的预测
"""
import re
import string
import pickle as pkl
from unicodedata import normalize
import numpy as np
from numpy.random import rand
from numpy.random import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


#加载数据到内存中
def load_doc(filename):
    file=open(filename,'rt',encoding='utf-8')
    #read all the data
    text=file.read()
    file.close()
    return text
#将加载的文件内容转变成单个的句子
def to_pairs(doc):
    lines=doc.strip().split('\n')
    pairs=[line.split('\t') for line in lines]
    return pairs
# clean a list of lines
def clean_pairs(lines):
    cleaned=list()
    #prepare regex for char filtering
    re_print=re.compile('[^%s]'%re.escape(string.printable))
    #prepare translation table for removing punctutation
    table=str.maketrans('','',string.punctuation)

    for pair in lines:
        clean_pair=list()
        for line in pair:
            #normalize unicode characters
            line=normalize('NFD',line).encode('ascii','ignore')
            line=line.decode('UTF-8')
            #使用空格进行分割
            line=line.split()
            #将每个单词转变成小写
            line=[word.lower() for word in line]
            #移除标点符号
            line=[word.translate(table) for word in line]
            #移除不可打印的字符
            line=[re_print.sub('',w) for w in line]
            #移除数值型token
            line=[word for word in line if word.isalpha()]
            #作为一个字符串存储
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)

#save a list of clean sentences to file
def save_clean_data(sentences,filename):
    pkl.dump(sentences,open(filename,'wb'))
    print('saved:%s'%filename)

#load a clean dataset
def load_clean_sentences(filename):
    return pkl.load(open(filename,'rb'))

#fit a tokenizer
def create_tokenizer(lines):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
#max sentences length
def max_length(lines):
    return max(len(line.split()) for line in lines)

#encode and pad sequences
def encode_sequences(tokenizer,length,lines):
    #integer encode sequence
    X=tokenizer.texts_to_sequences(lines)
    #pad sentences with 0 values
    X=pad_sequences(X,maxlen=length,padding='post')
    return X

# one-hot encode target sequencs
def encode_output(sequences,vocab_size):
    ylist=[]
    for sequence in sequences:
        encoded=to_categorical(sequence,num_classes=vocab_size)
        ylist.append(encoded)
    y=np.array(ylist)
    y=y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)
    return y

#define NMT model
def defind_model(src_vocab,tar_vocab,src_timesteps,tar_timesteps,n_units):
    model=Sequential()
    model.add(Embedding(src_vocab,n_units,input_length=src_timesteps,mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab,activation='softmax')))
    return model

#map an integer to a word
def word_for_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

#Generate target given source sequence
def predict_sequence(model,tokenizer,source):
    prediction=model.predict(source,verbose=0)[0]
    integers=[np.argmax(vector) for vector in prediction]
    target=[]
    for i in integers:
        word=word_for_id(i,tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model,tokenizer,sources,raw_dataset):
    actual,predicted=list(),list()
    for i,source in enumerate(sources):
        #translate encoded source text
        source=source.reshape(1,source.shape[0])
        translation=predict_sequence(model,eng_tokenizer,source)
        raw_target,raw_src=raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
        # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == '__main__':
    #加载数据集
    filename='deu-eng\deu.txt'
    doc=load_doc(filename)
    #split into english-german pairs
    pairs=to_pairs(doc)
    #clean sentences
    clean_pairs=clean_pairs(pairs)
    #save clean pairs to file
    save_clean_data(clean_pairs,'deu-eng\english-german.pkl')
    #spot check
    for i in range(10):
        print('[%s]->[%s]'%(clean_pairs[i,0],clean_pairs[i,1]))

    #加载数据集
    raw_dataset=load_clean_sentences('deu-eng\english-german.pkl')
    #reduce dataset size
    n_sentences=10000
    dataset=raw_dataset[:n_sentences,:]
    #random shuffle data
    shuffle(dataset)

    #split data into train/test
    train,test=dataset[:9000],dataset[9000:]
    #save
    save_clean_data(dataset,'deu-eng\english-german-both.pkl')
    save_clean_data(train,'deu-eng\english-german-train.pkl')
    save_clean_data(test,'deu-eng\english-german-test.pkl')

    #1.加载数据集
    dataset=load_clean_sentences('deu-eng\english-german-both.pkl')
    train=load_clean_sentences('deu-eng\english-german-train.pkl')
    test=load_clean_sentences('deu-eng\english-german-test.pkl')

    #2.Prepare english tokenizer
    eng_tokenizer=create_tokenizer(dataset[:,0])
    eng_vocab_size=len(eng_tokenizer.word_index)+1
    eng_length=max_length(dataset[:,0])
    print('English Vocabulary Size:%d'%eng_vocab_size)
    print('English Max Length:%d'%eng_length)

    # prepare german tokenizer
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('German Vocabulary Size: %d' % ger_vocab_size)
    print('German Max Length: %d' % (ger_length))

    #3. Prepare training data
    #德语-->英语
    trainX=encode_sequences(ger_tokenizer,ger_length,train[:,1])
    trainY=encode_sequences(eng_tokenizer,eng_length,train[:,0])
    trainY = encode_output(trainY, eng_vocab_size)
    #4. Prepare validation data
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)

    #4. Define model
    model=defind_model(ger_vocab_size,eng_vocab_size,ger_length,eng_length,256)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    #summarize defined model
    print(model.summary())
    #plot_model(model,to_file='model.png',show_shapes=True)
    #5. fit model
    checkpoint=ModelCheckpoint('deu-eng\model.h5',monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    model.fit(trainX,trainY,epochs=30,batch_size=64,validation_data=(testX,testY),callbacks=[checkpoint],verbose=2)

    # 6.评估模型表现
    # load model
    model = load_model('deu-eng\model.h5')
    # test on some training sequences
    print('train')
    evaluate_model(model, eng_tokenizer, trainX, train)
    # test on some test sequences
    print('test')
    evaluate_model(model, eng_tokenizer, testX, test)







