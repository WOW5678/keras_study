# -*- coding:utf-8 -*-
'''
function:
save model and reload model
'''
import numpy as np
np.random.seed(1337)

from keras.models import  Sequential
from keras.layers import Dense
from keras.models import load_model

# Create some data
# X是一个行向量 200个元素
x=np.linspace(-1,1,200)
print(x)
np.random.shuffle(x)
y=0.5*x+np.random.normal(0,0.05,(200,))
x_train,y_train=x[:160],y[:160]
x_test,y_test=x[160:],y[160:]
model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))
model.compile(loss='mse',optimizer='sgd')
for step in range(300):
    cost=model.train_on_batch(x_train,y_train)

# save
print('test before save:',model.predict(x_test[:2]))
model.save('my_model.h5')
del model

# load
model=load_model('my_model.h5')
print('test after save:',model.predict(x_test[:2]))

# save and reload the weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

# Save and load fresh network without trained weights
from keras.models import  model_from_json
json_string=model.to_json()
model=model_from_json(json_string)