# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:24:08 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

#处理训练集
train_data = pd.read_csv(
    "./data/train.csv", lineterminator='\n',encoding='utf-8')

train_data.columns
train_data.drop(
        ['ID'],
        axis=1,
        inplace=True
)

def GetLabel(x):
    
    '''得到训练集label'''
    
    if x == 'Positive':
        return 1
    else:
        return 0
    
train_data['label'] = train_data.label.apply(GetLabel)

def GetVocabulary(vocab_s,mode='train'):
    
    '''
        mode:'train'
            处理训练集去除非字母后返回语料库的字典和语料库的列表格式
        mode:None
            处理测试集去除非字母后返回语料库的列表格式
    '''
    
    vocab_list = []
    if mode == 'train':
        vocabs = []
        for i in vocab_s:
            tokens = word_tokenize(i)
            vocab_ = []
            for j in tokens:
                if j.isalpha():
                    vocab_.append(j)
                    vocabs.append(j)
            vocab_list.append(vocab_)
        vocabs = sorted(set(vocabs))
        return {u: i for i, u in enumerate(vocabs)},vocab_list
    else:
        for i in vocab_s:
            tokens = word_tokenize(i)
            vocab_ = []
            for j in tokens:
                if j.isalpha():
                    vocab_.append(j)
            vocab_list.append(vocab_)
        return vocab_list
dict_voc,vocab_list = GetVocabulary(train_data.review,mode='train')

def TextHelper(x,keys):
    
    '''文本帮助函数:映射字典处理语料库'''
    
    text_int = []
    for i in x:
        if i in keys:
            text_int.append(dict_voc.get(i))
        else:
            text_int.append(0)
    return text_int

def TextAsInt(data,text_list,mode='train'):
    
    '''将文本转换为向量化'''
    
    char_dataset = []
    keys = list(tuple(dict_voc.keys()))
    if mode == 'train':
        label = data.label
        for i in text_list:
            char_dataset.append(TextHelper(i,keys))
        return char_dataset,label.values
    else:
        for i in text_list:
            char_dataset.append(TextHelper(i,keys))
        return char_dataset
char_dataset,label = TextAsInt(train_data,vocab_list,mode='train')    
#char_dataset = tf.data.Dataset.from_tensor_slices(char_dataset.values[0])

#BUFFER_SIZE = 10000
#BATCH_SIZE = 64
#train_dataset = char_dataset.shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.batch(BATCH_SIZE)


#模型设置
maxlen = 400
char_dataset = keras.preprocessing.sequence.pad_sequences(char_dataset,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=maxlen)
#模型搭建
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(dict_voc)+1,16,input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#切分数据集
x_train,x_test,y_train,y_test = train_test_split(char_dataset,label,test_size=0.3,random_state=0)

# 训练模型
history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2)

model.evaluate(x_test,y_test)



#处理测试集
test_data = pd.read_csv(
    "./data/test.csv", lineterminator='\n')
test_id = test_data.ID.values
test_data.drop(['ID'],axis=1,inplace=True)
test_vocab_list = GetVocabulary(test_data.review,mode=None)
test_char_dataset = TextAsInt(test_data,test_vocab_list,mode=None)  

def PadToSize(vec, size):
    
    '''填充文本向量'''
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

pad = True
def SamplePredict(sentence, pad):
    
    '''预测单条数据'''
    
    #tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)
    tokenized_sample_pred_text = sentence
    if pad:
        tokenized_sample_pred_text = PadToSize(tokenized_sample_pred_text, maxlen)
  
    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return predictions

#SamplePredict(test_char_dataset[1], pad)
predictions = []
for i in range(len(test_char_dataset)):
    print("正在预测第%d条记录,总计有%d条记录"%(i+1,len(test_char_dataset)))
    prediction = SamplePredict(test_char_dataset[i], pad)
    predictions.append(prediction[0][0])

#保存提交文件
submit = pd.DataFrame()
submit['ID'] = test_id
submit['Pred'] = predictions
submit.to_csv("./data/submit.csv",index=False)
