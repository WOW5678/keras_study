# -*- coding:utf-8 -*-
'''
sequence to sequence example in keras(character-level)

This script demonstarate how to implement a basic character-level sequence-sequence model
We apply it to translating short english sentence to French sentence, character-by-character.
Note that it's unusual to do character-level machine translation，as word-level models are more
common in this domain.
'''

import numpy as np
from keras.models import Model
from keras.layers import Input,Dense,LSTM

# Parameters
batch_size=64
epochs=100
latent_dim=256
num_samples=10000
data_path='data/fra-eng/fra.txt'

# Vectorize the data
input_texts=[]
target_texts=[]
input_characters=set()
target_charaters=set()
with open(data_path,'r',encoding='utf-8') as f:
    lines=f.read().split('\n')
# 为什么减1呢？？？？
for line in lines[:min(num_samples,len(lines))]:
    input_text,target_text=line.split('\t')
    # We use 'tab' as the start sequence character
    # for the targets, and '\n' as end sequence character
    target_text='\t'+target_text+'\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_charaters:
            target_charaters.add(char)

input_characters=sorted(list(input_characters))
target_charaters=sorted(list(target_charaters))
num_encoder_tokens=len(input_characters)
num_decoder_tokens=len(target_charaters)
# 找出input_texts中最长句子的长度
max_encoder_seq_len=max([len(txt) for txt in input_texts])
# 找出目标句子中最长句子的长度
max_decoder_seq_len=max([len(txt) for txt in target_texts])

print('Number of samples:',len(input_texts))
print('Number of unique input tokens:',num_encoder_tokens)
print('Number of unique output tokens:',num_decoder_tokens)
print('Max sequence length for inputs:',max_encoder_seq_len)
print('Max sequence length of outputs:',max_decoder_seq_len)

input_token_index=dict([(char,i) for i, char in enumerate(input_characters)])
target_token_index=dict([(char,i) for i,char in enumerate(target_charaters)])
# onehot 编码
encoder_input_data=np.zeros((len(input_texts),max_encoder_seq_len,num_encoder_tokens),dtype='float32')
decoder_input_data=np.zeros((len(input_texts),max_decoder_seq_len,num_decoder_tokens),dtype='float32')
# len(input_texts)与len(target_texts)是相等的
decoder_target_data=np.zeros((len(target_texts),max_decoder_seq_len,num_decoder_tokens),dtype='float32')

for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i,t,input_token_index.get(char)]=1
    for t,char in enumerate(target_text):
        decoder_target_data[i,t,target_token_index.get(char)]=1
        if t>0:
            #decoder_target_data will be ahead by one timestep
            # and will not include the start characters
            # 前一行（即前一个时刻）的同一个位置为1
            decoder_target_data[i,t-1,target_token_index.get(char)]=1
# Define an input sequence and process it
encoder_inputs=Input(shape=(None,num_encoder_tokens))
encoder=LSTM(latent_dim, return_state=True)
encoder_outputs,state_h,state_c=encoder(encoder_inputs)
# We discard encoder outputs and only keep the states
encoder_states=[state_h,state_c]

# set up the decoder using encoder_state as initial state
decoder_inputs=Input(shape=(None,num_decoder_tokens))
decoder_lstm=LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=True)
decoder_dense=Dense(num_decoder_tokens,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)
# Define the model that will return
model=Model([encoder_inputs,decoder_inputs],decoder_outputs)

# Run training
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
model.fit([encoder_input_data,decoder_target_data],decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)

#save model
model.save('s2s.h5')

# Next:inference mode(sampling)
#Here is the drill:
# 1) encode input and retrieve initial decoder state
# 2)run one step of decoder with this initial state
# and a start of sequence token as target
# output will be the next taret token
# 3)repeat with the current target token and current states

# define sampling samples
encoder_model=Model(encoder_inputs,encoder_states)
decoder_state_input_h=Input(shape=(latent_dim,))
decoder_state_input_c=Input(shape=(latent_dim,))
decoder_state_inputs=[decoder_state_input_h,decoder_state_input_c]
decoder_outputs,state_h,state_c=decoder_lstm(decoder_inputs,initial_state=decoder_state_inputs)
decoder_states=[state_h,state_c]
decoder_model=Model([decoder_inputs]+decoder_state_inputs,[decoder_outputs]+decoder_states)

# Reverse lookup token index to decode sequence back to
# something readable
reverse_input_char_index=dict((i,char) for char,i in input_token_index.items())
reverse_target_char_index=dict((i,char) for char,i in target_token_index.items())

def decoder_sequence(input_seq):
    # Encode the input as state vector
    states_values=encoder_model.predict(input_seq)
    # generate empty target sequence of length 1
    target_seq=np.zeros((1,1,num_decoder_tokens))
    # populate the first charater of target sequence with the start character
    target_seq[0,0,target_token_index['\t']]=1

    # Sampling loop for a batch of sequence
    stop_condition=False
    decoder_sentense=''
    while not stop_condition:
        output_tokens,h,c=decoder_model.predict([target_seq]+states_values)

        #Sample a token
        sampled_token_index=np.argmax(output_tokens[0,-1,:])
        sampled_char=reverse_target_char_index[sampled_token_index]
        decoder_sentense+=sampled_char

        if (sampled_char=='\n' or len(decoder_sentense)>max_decoder_seq_len):
            stop_condition=True
        # update states
        states_values=[h,c]
    return decoder_sentense

for seq_index in range(100):
    input_seq=encoder_input_data[seq_index:seq_index+1]
    decoder_sentence=decoder_sentence(input_seq)
    print('---')
    print('Input sentence:',input_texts[seq_index])
    print('Decoded sentence:',decoder_sentence)
