#导入库和要处理的数据集
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.src.callbacks import ModelCheckpoint
#文本预处理
#import nltk
#nltk.download()
import jieba
import jieba.analyse
#文本转数字
import  keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray
from numpy import zeros
#构造模型
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
#训练
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, f1_score

train_df =pd.read_csv('train.news.csv',encoding='utf-8')
#新闻网址、图片网址和评价对识别没有帮助，所以删除这些列
train_df=train_df.drop(['News Url','Image Url','Report Content'],axis=1)#
print(train_df.head(6))
print(train_df.shape)
print(train_df.isnull().sum())                           #检查是否存在缺失值
train_df=train_df.dropna()                                     #删除缺行
print(train_df.shape)
train_df=train_df.sample(frac=1).reset_index(drop=True)        #对数据集进行打乱     train_df = shuffle(train_df).reset_index(drop=True)
print(train_df.head(6))
sns.countplot(train_df,x='label')

punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
stop_words=[words.strip() for words in open('stop_words.txt',encoding='utf-8').readlines()]
def cleaning(text): #清洗函数
    cutwords=list(jieba.lcut_for_search(text))
    final_cutwords=''
    for words in cutwords:
        if words not in (punc and stop_words):
            final_cutwords+=(words+' ')
    return final_cutwords

#分词并完成清洗
t=pd.DataFrame(train_df.astype(str))
print(t.head())
train_df['text']=t['Title']+t['Ofiicial Account Name']#+t['Report Content']
t=pd.DataFrame(train_df.astype(str))
train_df['text']=t['text'].apply(cleaning)
texts=train_df['text']
print(texts.head())
targets=np.asarray(train_df['label'])

#文本转数字
Num_words=20100                                                    #定义最大词汇量和序列长度
tokenize=Tokenizer(num_words=Num_words,oov_token='<OOV>')    #文本转序列，处理未知单词，并保持序列长度的一致性
tokenize.fit_on_texts(texts)
print(tokenize.document_count)
vocab_size=len(tokenize.word_index)+1
print("wordsnumber：",vocab_size)
sequences=tokenize.texts_to_sequences(texts)

#确认maxlen的合理区间
# 计算每个文本的长度
text_lengths = [len(text.split()) for text in train_df['text']]

# 将列表转换为 Pandas Series
text_lengths_series = pd.Series(text_lengths)
# 计算每个长度的文本数量
length_counts = text_lengths_series.value_counts().sort_index()
# 计算累计百分比
cumulative_percentage = length_counts.cumsum() / length_counts.sum() * 100
# 绘制文本长度的直方图
plt.hist(text_lengths, bins=30, edgecolor='k', alpha=0.7)
plt.title('Text Length Distribution')
plt.xlabel('Length of Text')
plt.ylabel('Number of Texts')
plt.show()

# 绘制累计百分比图
cumulative_percentage.plot()
plt.title('Cumulative Percentage of Text Length')
plt.xlabel('Text Length')
plt.ylabel('Cumulative Percentage')
plt.grid(True)
plt.show()
'''
# 定义长度区间
bins = [0, 5, 10, 15,20, 25, 30, 100, 500, 1000]   #根据结果自己调整
# 使用 pd.cut 计算各个区间的数量
length_distribution = pd.cut(text_lengths, bins, right=False).value_counts().sort_index()
# 计算百分比
length_distribution_percentage = (length_distribution / length_distribution.sum()) * 100
# 打印各区间百分比
print(length_distribution_percentage)
# 可视化结果
length_distribution_percentage.plot(kind='bar')
plt.title('Text Length Distribution Percentage')
plt.xlabel('Text Length Range')
plt.ylabel('Percentage (%)')
plt.show()
'''
# 计算合适的max_len
# 比如选择覆盖95%的文本长度
percentile = 95
print(f"选择覆盖 {percentile}% 文本的长度值 max_len: {int(np.percentile(text_lengths, percentile))}")

max_len=50                                                         # 定义最大词汇量和序列长度
text_data=pad_sequences(sequences, maxlen=max_len,padding="post", truncating="post")     #padding，truncating 超出时从后面截取 不足时从后面补0
print(text_data)
one_hot_matrix=tokenize.texts_to_matrix(texts,mode="binary")        #文本转矩阵
print(one_hot_matrix.shape)

embedding_index=dict()
f=open('vectors.txt',encoding='utf-8')
for line in f:
    values=line.split()
    word=values[0]
    coefs=asarray(values[1:],dtype='float32')
    embedding_index[word]=coefs
f.close()
print("load %s word vectors"%len(embedding_index))

#为训练数据集中的每个单词创建一个嵌入矩阵
#我们可以通过枚举Tokenizer.word_index中的所有唯一单词并从加载的GloVe嵌入中找到嵌入权重向量来实现这一点
'''
vectors.txt=glove.6B.50d(维度）
'''
embedding_matrix=zeros((vocab_size,50))              #初始化嵌入式矩阵，矩阵的行数等于词汇表大小（vocab_size），列数（纬度）为50
for word,i in tokenize.word_index.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

#构造模型

model=Sequential()
#嵌入层，用来降维
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len, weights=[embedding_matrix], trainable=False  ))
model.add(Bidirectional(LSTM(units=64,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=64,return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile( loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(0.0012),#学习率
                metrics='accuracy')
print(model.summary())

#进行训练

x_train,x_test,y_train,y_test=train_test_split(text_data, targets, test_size=0.2,random_state=20,  shuffle=True)
cp=ModelCheckpoint('model_Rnn.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='auto')
history=model.fit(text_data,targets,epochs=10,batch_size=32,shuffle=True, validation_split=0.2, validation_freq=1, callbacks=cp)
#绘图

#测试
test_df=pd.read_csv('test.feature.csv',encoding='utf-8')
submit_example = pd.read_csv('submit_example.csv')
test_df=test_df.drop(['News Url','Image Url','Report Content'],axis=1)
print(test_df.shape)
print(test_df.head())
print(test_df.isnull().sum())
#test_df=test_df.dropna()
#test_df=test_df.sample(frac=1).reset_index(drop=True)
print(test_df.head())
tt=pd.DataFrame(test_df.astype(str))
test_df['text']=tt['Title']+tt['Ofiicial Account Name']#+tt['Report Content']
test_df['text']=test_df['text'].apply(cleaning)
test_text=test_df['text']
print(test_text)

test_sequence=tokenize.texts_to_sequences(test_text)
test_data=pad_sequences(test_sequence, maxlen=max_len,  padding='post',truncating='post')
preds = model.predict(test_data)
predictions = (preds >= 0.5).astype(int).flatten()
print(preds)
result=list()
for i in range(len(preds)):
    result.append(preds[i])
'''
# 计算评估指标
test_accuracy = accuracy_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1_score = f1_score(y_test, y_pred)

print("Test Accuracy: ", test_accuracy)
print("Test Recall: ", test_recall)
print("Test F1 Score: ", test_f1_score)
'''

submission = pd.DataFrame({'id': submit_example['id'], 'label': predictions})
submission.to_csv('submit_example_bilstm.csv', index=False)
