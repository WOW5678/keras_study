# -*- coding:utf-8 -*-
'''
有时候我们需要的并不是最后的结果，而是中间某一层的结果，这个例子演示如何得到中间结果
'''
from keras.layers import Dense,Activation,Input
from keras.models import  Sequential,Model
import  numpy as np

x=np.random.standard_normal(size=(200,12))
y=np.random.randint(low=0,high=2,size=(200,1))
# this is your initial model
input=Input(shape=(12,))
hidden_1=Dense(20,activation='tanh')(input)
hidden_2=Dense(64,activation='tanh',name='hidden_2')(hidden_1)
hidden_3=Dense(1,activation='softmax')(hidden_2)
model=Model(input=input,output=hidden_3)

# we train it
model.compile(loss='mse',optimizer='sgd')
model.fit(x,y,epochs=20,batch_size=10)

# we build a new model with the activations of old model
# this model is truncted after the first layer
test_x=np.random.standard_normal(size=(1,12))

model2= Model(inputs=model.input, outputs=model.get_layer('hidden_2').output)
activations=model2.predict(test_x)
print(activations)
