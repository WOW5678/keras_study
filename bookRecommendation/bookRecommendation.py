# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/14 0014 下午 9:18
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 实现一个简单的书籍推荐系统
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from keras.layers import Input,Embedding,Flatten,Dot,Dense
from keras.models import Model
from keras.utils.vis_utils import  model_to_dot

import warnings
warnings.filterwarnings('ignore')

#加载数据集
dataset=pd.read_csv('ratings.csv')
#   book_id  user_id  rating
# 0        1      314       5
# 1        1      439       3
print(dataset.head(5))
print(dataset.shape) #(981756, 3)

#随机分割训练集和测试集
from sklearn.model_selection import train_test_split
train,test=train_test_split(dataset,test_size=0.2,random_state=42)

#统计共有多少本书 多少个用户
n_users=len(dataset.user_id.unique())
n_books=len(dataset.book_id.unique())
#53424 10000
print(n_users,n_books)

#创建模型
book_input=Input(shape=[1],name='book-input')
book_embedding=Embedding(n_books+1,5,name='book-embedding')(book_input)
book_vec=Flatten(name='flatten-books')(book_embedding)

user_input=Input(shape=[1],name='user-input')
user_embedding=Embedding(n_users+1,5,name='user-embedding')(user_input)
user_vec=Flatten(name='flatten-user')(user_embedding)

#书籍向量和用户向量进行点乘（求匹配得分）
prob=Dot(name='dot-product',axes=1)([book_vec,user_vec])
model=Model([user_input,book_input],prob)
model.compile('adam','mean_squared_error')

from keras.models import load_model
if os.path.exists('regression_model.h5'):
    model=load_model('regression_model.h5')
else:
    history=model.fit([train.user_id,train.book_id],train.rating,epochs=1,verbose=1)
    model.save('regression_model.h5')
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')

model.evaluate([test.user_id,test.book_id],test.rating)
predicitons=model.predict([test.user_id.head(10),test.book_id.head(10)])
#打印每个样本的得分和真实的rating
[print(predicitons[i],test.rating.iloc[i]) for i in range(10)]


#可视化embedding
from sklearn.decomposition import PCA
import seaborn as sns

#提取embedding
book_emb=model.get_layer('book-embedding')
book_emb_weights=book_emb.get_weights()[0]
print(book_emb_weights.shape)

pca=PCA()
pca_result=pca.fit_transform(book_emb_weights)
sns.scatterplot(x=pca_result[:,0],y=pca_result[:,1])
plt.show()

book_emb_weights=book_emb_weights/np.linalg.norm(book_emb_weights,axis=1).reshape((-1,1))

pca=PCA(n_components=2)
pca_result=pca.fit_transform(book_emb_weights)
sns.scatterplot(x=pca_result[:,0],y=pca_result[:,1])
plt.show()

from sklearn.manifold import TSNE
tsne=TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
tsne_results=tsne.fit_transform(book_emb_weights)

sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1])
plt.show()
#最后一步 做推荐
book_data=np.array(list(set(dataset.book_id)))
user=np.array([1 for i in range(len(book_data))])
predicitons=np.array([a[0] for a in predicitons])
#argsort()返回排序索引，默认是从小到大
recommended_book_ids=(-predicitons).argsort()[:5]
#打印预测的得分
print(predicitons[recommended_book_ids])

#将对应的book_id与book对应起来
books=pd.read_csv('books.csv')
print(books.head(5))
print(books[books['id'].isin(recommended_book_ids)])

