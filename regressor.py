# -*- coding:utf-8 -*-
'''
模拟回归问题
'''
import numpy as np
np.random.seed(100)
from keras.models import  Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
# Create some data
X=np.linspace(-1,1,200)
np.random.shuffle(X)
Y=0.5*X+np.random.normal(0,0.05,(200,))

# Plot data
plt.scatter(X,Y)
plt.show()

X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]

# Build model
# Activate function
model=Sequential()
model.add(Dense(output_dim=1,input_dim=1))

model.compile(loss='mse',optimizer='sgd')

# Train the model
print('Training.....')
for step in range(301):
    cost=model.train_on_batch(X_train,Y_train)
    if step%100==0:
        print('Train Cost:',cost)

# Test model
print('Testing...')
cost=model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
w,b=model.layers[0].get_weights()
print('Weights=%s,biases=%s'%(w,b))

# Virualize the result
# plot the prediction
Y_pred=model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()
