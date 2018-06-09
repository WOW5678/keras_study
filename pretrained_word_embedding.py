# -*- coding:utf-8 -*-
'''
This script loads pre-trained word embeddings(Glove embeddings) into a forzen keras
embedding layer, and use it to train a text classification model on  the
20 newsgroup dataset(classification of newgroup message into 20 different categroies)

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html

'''
from __future__ import print_function
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense,Input,GlobalMaxPool1D
from keras.layers import Conv1D,MaxPooling1D,Embedding
from keras.models import Model

base_dir='data'
glove_dir=os.path.join(base_dir,'glove.6B')
text_data_dir=os.path.join(base_dir,'news20\\20_newsgroup')
max_sequence_lenght=200
max_num_words=20000
embedding_dim=100
validation_split=0.2

# First, build index mapping words in the embedding set to their embedding vector
print('Indexing word vector')

embedding_index={}
with open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype='float32')
        embedding_index[word]=coefs
print('Found %s word vector.'%len(embedding_index))

# Second prepare text sample and labels
print('prepare text dataset')
texts=[]
labels_index={}
labels=[]
for name in sorted(os.listdir(text_data_dir)):
    path=os.path.join(text_data_dir,name)
    if os.path.isdir(path):
        label_id=len(labels_index)
        labels_index[name]=label_id
        for filename in sorted(os.listdir(path)):
            if filename.isdigit():
                fpath=os.path.join(path,filename)
                args={} if sys.version_info<(3,)else{'encoding':'latin-1'}
                with open(fpath,**args) as f:
                    t=f.read()
                    i=t.find('\n\n')
                    if 0<i:
                        t=t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts'%len(texts))
# Finally ,vectorize the text samples into 2d integer tensor
tokenizer=Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
print('sequences:',sequences)
word_index=tokenizer.word_index
print('word_index:',word_index)
print('Found %s unique tokens.'%len(word_index))

data=pad_sequences(sequences,maxlen=max_sequence_lenght)
print('data:',data)
labels=to_categorical(np.asarray(labels))
print('shape of data shape:',data.shape)
print('shape of labels:',labels.shape)
# Split the data into a traing dataset and validation dataset
indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]

num_validation_samples=int(validation_split*data.shape[0])

x_train=data[:-num_validation_samples]
y_train=labels[:-num_validation_samples]
x_val=data[-num_validation_samples:]
y_val=labels[-num_validation_samples:]

print('Prepare embedding matrix.')
num_words=min(max_num_words,len(word_index)+1)
embedding_matrix=np.zeros((num_words,embedding_dim))
for word,i in word_index.items():
    if i>=max_num_words:
        continue
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

# load pre-trained word embedding into an Embedding layer
# note that we set trainable =False so as to keep the embedding fix
embedding_layer=Embedding(num_words,embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_sequence_lenght,
                          trainable=False)
print('Train Model')

# Train a 1D convet with global maxpooling
sequence_input=Input(shape=(max_sequence_lenght,),dtype='int32')
embedding_sequences=embedding_layer(sequence_input)
x=Conv1D(128,3,activation='relu')(embedding_sequences)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPool1D()(x)
x=Dense(128,activation='relu')(x)
pred=Dense(len(labels_index),activation='softmax')(x)

model=Model(sequence_input,pred)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_val,y_val))