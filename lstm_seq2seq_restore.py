# -*- coding:utf-8 -*-
''''

'''
import keras
from keras.layers import Dense,Input,LSTM
from keras.models import Sequential,load_model,Model
import numpy as np

batch_size=64
epochs=100
latent_dim=256
num_samples=10000
data_path='data/fra-eng/fra.txt'

input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()
with open(data_path,'r',encoding='utf-8') as f:
    lines=f.read().split('\n')
for line in lines[:min(num_samples,len(lines)-1)]:
    input_text,target_text=line.split('\t')
    target_text='\t'+target_text+'\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        input_characters.append(char)
    for char in target_text:
        target_characters.append(char)
input_characters=set(list(input_characters))
target_characters=set(list(target_characters))
num_encoder_token=len(input_characters)
num_decoder_token=len(target_characters)
max_encoder_seq_length=max([len(text) for text in input_texts])
max_decoder_seq_length=max([len(text) for text in target_texts])
print('Number of samples:',len(input_texts))
print('Number of unique input token:',len(input_characters))
print('Number of unique target tokens:',len(target_characters))
print('Max sequence length for input:',max_encoder_seq_length)
print('Max sequence length for target:',max_decoder_seq_length)

input_token_index=dict((c,i) for i,c in enumerate(input_characters))
target_token_index=dict((c,i) for i,c in enumerate(target_characters))

encoder_input_data=np.zeros((len(input_texts),max_encoder_seq_length,num_encoder_token),dtype='float32')
for i,input_text in enumerate(input_texts):
    for t,char in input_text:
        encoder_input_data[i,t,input_token_index[char]]=1
# Restore the model and construct the encoder and decoder
model=load_model('s2s.h5')
encoder_inputs=model.input[0] # input_1
encoder_outputs,state_h_enc,state_c_enc=model.layers[2].output #lstm_1
encoder_states=[state_h_enc,state_c_enc]
encoder_model=Model(encoder_inputs,encoder_states)

decoder_inputs=model.input[1]
decoder_state_h=Input(shape=(latent_dim,),name='input_3')
decoder_state_c=Input(shape=(latent_dim,),name='input_4')
decoder_states_inputs=[decoder_state_h,decoder_state_c]
decoder_lstm=model.layers[3]
decoder_outputs,state_h_dec,state_c_dec=decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)
decoder_states=[state_h_dec,state_c_dec]
decoder_dense=model.layers[4]
decoder_outputs=decoder_dense(decoder_outputs)
decoder_model=Model([decoder_inputs]+decoder_states_inputs,[decoder_outputs]+decoder_states)

# Reverse lookup token index to decode sequences back to something readable
reverse_input_char_index=dict((i,char) for char,i in input_token_index.items())
reverse_target_char_index=dict((i,char) for char,i in target_token_index.items())

# Decode an input sequences. Future works should support beam search
def decode_sequnence(input_seq):
    # Eencoder the input as state vector
    state_value=encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq=np.zeros((1,1,num_decoder_token))
    # Populate the first character of taret sequence with the start sequences
    target_seq[0,0,target_token_index['\t']]=1

    # Sampling loop for a batch of sequence
    # to simplify,here we assume a batch of size 1
    stop_condition=False
    decoded_sequence=''
    while not stop_condition:
        output_token,h,c=decoder_model.predict([target_seq]+state_value)
        # sample a token
        sampled_token_index=np.argmax(output_token[0,-1,:])
        sampled_char=reverse_target_char_index[sampled_token_index]
        decoded_sequence+=sampled_char

        # Exit condiction:either hit max length or find stop character
        if (sampled_char =='\n') or len(decoded_sequence)>max_decoder_seq_length:
            stop_condition=True
        # Update the target sequence (of length 1)
        target_seq=np.zeros((1,1,num_decoder_token))
        target_seq[0,0,sampled_token_index]=1

        # update the state
        state_value=[h,c]

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding
    input_seq=encoder_input_data[seq_index:seq_index+1]
    decoded_sentence=decode_sequnence(input_seq)
    print('--')
    print('Input sentence:',input_texts[seq_index])
    print('Decoded sentence:',decoded_sentence)
