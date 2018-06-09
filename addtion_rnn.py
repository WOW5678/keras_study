# -*- coding:utf-8 -*-
'''
An implementation of sequence to sequence learning for performing addition
input:'535+61'
output:'596'

seq2seq的结构，encoder的输出（隐藏层的输出）复制digits+1份作为decoder的输入

'''
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

class CharacterTable(object):
    '''
    Given a set of characters:
    Encode them to a one hot integer representation
    Decode the one hot integer repersentation to their character output
    Decode a vector of probabilities to their character output
    '''
    def __init__(self,chars):
        """
        Initialize character table
        :param chars: characters that can appear in the input
        """
        self.chars=sorted(set(chars))
        self.char_indices=dict((c,i) for i,c in enumerate(self.chars))
        self.indices_char=dict((i,c) for i,c in enumerate(self.chars))

    def encode(self, C,num_rows):
        """
        one hot encode given string C
        :param C: a string
        :param num_rows:  how many first characters to consider in the string C
        :return:  encoded matrix of string C
        """
        x=np.zeros((num_rows,len(self.chars)))
        for i,c in enumerate(C):
            x[i,self.char_indices[c]]=1
        return x
    
    def decode(self,x,calc_argmax=True):
        '''
        :param x: an encoded matrix of string
        :param calc_argmax: bool if false do nothing
        :return: decoded string
        '''
        if calc_argmax:
            #把每行最大的那个概率对应的id(也是char的id)得到
            x=x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors(object):
    # 定义不同的颜色
    ok='\033[92m'
    fail='\033[91m'
    close='\033[0m'

# Parameters for the model and dataset
Training_size=50
Digits=3
Revererse=True

# Maximum length of input
# 输入是一个字符串,形如“123+456”
maxlen=Digits+1+Digits

# All the number, plus sign and space for padding
chars='0123456789+ '
ctabel=CharacterTable(chars)

questions=[]
expected=[]
seen=set()
print('Generating data...')
while len(questions)<Training_size:
    f=lambda:int(''.join(np.random.choice(list('0123456789'))
                           for i in range(np.random.randint(1,Digits+1))))
    a,b=f(),f()
    print('a:',a)
    print('b:',b)
    # skip any addition question we've already seen
    # Also skip any such that x+y==y+x(hence the sorting)
    key=tuple(sorted((a,b)))
    if key in seen:
        continue
    seen.add(key)

    # Pad the data with spaces such that it's always maxlen
    q='{}+{}'.format(a,b)
    query=q+' '*(maxlen-len(q))
    ans=str(a+b)
    # Answers can be of maximum size
    ans+=' '*(Digits+1-len(ans))
    ##???? 为什么要进行反转？只反转query 则ans的值与query 就不对应了？？
    print('query_before:',query)
    if Revererse:
        # Revise the query,e.g,"12+345" becomes"543+21", note the space used for padding
        query=query[::-1]
    print('query_after:',query)
    questions.append(query)
    expected.append(ans)
print('Total addition questions:',len(questions))

print('Vecotorization.....')
x=np.zeros((len(questions),maxlen,len(chars)),dtype=np.bool)
y=np.zeros((len(questions),Digits+1,len(chars)),dtype=np.bool)
for i,sentence in enumerate(questions):
    print('sentence:',repr(sentence))
    x[i]=ctabel.encode(sentence,maxlen)
for i, sentence in enumerate(expected):
    y[i]=ctabel.encode(sentence,Digits+1)

# Shuffle(x,y) in unison as the later parts of x will almost all be large digits
indices=np.arange(len(y))
np.random.shuffle(indices)
x=x[indices]
y=y[indices]

# Explicitiy set apart 10% for validation data that we never train over
split_data=len(x)-int(len(x)*0.1)
(x_train,x_val)=x[:split_data],x[split_data:]
(y_train,y_val)=y[:split_data],y[split_data:]
print('Training Data:')
print('X_train.shape:',x_train.shape)
print('y_train.shape:',y_train.shape)
print('Validation Data:')
print('x_val.shape:',x_val.shape)
print('y_val.shape:',y_val.shape)

# Try replacing GRU,OR simpleRNN
RNN=layers.LSTM
hidden_size=128
batch_size=128
LAYERS=1

print('Build the model.....')
model=Sequential()
# Encode the input sequence using an RNN,producing an output of hidden_size
# NOTE: in a situation, your input sequence  have a variabel length
# use input_shape=(None,num_features)
model.add(RNN(hidden_size,input_shape=(maxlen,len(chars))))
# As the decode RNN'S input,repeatedly provide with the last hidden state of RNN for
# each time step. Repeat 'DIGITS+1' times as that's the maximum lenght of output
model.add(layers.RepeatVector(Digits+1))
# The decode RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # but all the outputs so far in the form of (num_samples,timesteps,output_dim)
    # This is necessary as timeDistributed in the below expects the first dimension to be the timesteps.
    model.add(RNN(hidden_size,return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Train the model each generation ans show predictions against the validation dataset
for iteration in range(1,200):
    print('-'*50)
    print('Iteration',iteration)
    model.fit(x_train,y_train,batch_size=batch_size,epochs=1,validation_data=(x_val,y_val))

    #select 10 samples form the validation set at random so we can visualize errors
    for i in range(10):
        ind=np.random.randint(0,len(x_val))
        rowx,rowy=x_val[np.array([ind])],y_val[np.array([ind])]
        #print('rowx:',rowx)
        preds=model.predict_classes(rowx,verbose=0)
        print('preds:',preds)
        q=ctabel.decode(rowx[0])
        correct=ctabel.decode(rowy[0])
        guess=ctabel.decode(preds[0],calc_argmax=False)
        print('Q:',q[::-1] if Revererse else q,end=' ')
        print('T:',correct,end=' ')
        if correct==guess:
            print(colors.ok+'**'+colors.close,end=' ')
        else:
            print(colors.fail+'--'+colors.close,end=' ')
        print('guess:',guess)
