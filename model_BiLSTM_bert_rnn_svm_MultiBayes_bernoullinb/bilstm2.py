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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, recall_score, f1_score

train_df =pd.read_csv('train1.csv',encoding='utf-8')
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

# 统计每条 content 的汉字数量
train_df['char_count'] = train_df['content'].apply(len)
# 查看每条内容的汉字数量
print(train_df[['content', 'char_count']].head())
# 找出最长的一条的汉字数量
max_char_count = train_df['char_count'].max()
print(f"最长的一条 content 的汉字数量: {max_char_count}")
# 定义不同的百分位数
percentiles = [50, 75, 90, 95, 99]

# 计算并打印每个百分位数下的长度值
for p in percentiles:
    length_at_percentile = int(np.percentile(train_df['char_count'], p))
    print(f"覆盖 {p}% 的 content 长度: {length_at_percentile}")

#分词并完成清洗
t=pd.DataFrame(train_df.astype(str))
print(t.head())
train_df['text']=t['content']
t=pd.DataFrame(train_df.astype(str))
train_df['text']=t['text'].apply(cleaning)
texts=train_df['text']
print(texts.head())
targets=np.asarray(train_df['label'])

# 计算每条文本的长度
train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))

# 查看长度分布
print(train_df['text_length'].describe())

#文本转数字
Num_words=43510                                                    #定义最大词汇量和序列长度
tokenize=Tokenizer(num_words=Num_words,oov_token='<OOV>')    #文本转序列，处理未知单词，并保持序列长度的一致性
tokenize.fit_on_texts(texts)
print(tokenize.document_count)
vocab_size=len(tokenize.word_index)+1
print("number：",vocab_size)
sequences=tokenize.texts_to_sequences(texts)

# 统计词汇总数
total_vocabulary_size = len(tokenize.word_index)+1
print(f"数据集中的总词汇量: {total_vocabulary_size}")

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
max_len = int(np.percentile(train_df['text_length'], percentile))
print(max_len)
print(f"选择覆盖 {percentile}% 文本的长度值 max_len: {max_len}")                                                      # 定义最大词汇量和序列长度
text_data=pad_sequences(sequences, maxlen=max_len,padding="post", truncating="post")     #padding，truncating 超出时从后面截取 不足时从后面补0
print(text_data)
one_hot_matrix=tokenize.texts_to_matrix(texts,mode="binary")        #文本转矩阵
print(one_hot_matrix.shape)

embedding_index=dict()#使用预训练词嵌入来增强自然语言处理（NLP）模型
f=open('glove.txt',encoding='utf-8')
for line in f:                 #从词嵌入文件中加载词向量，并将它们存储在一个字典中。
    values=line.split()
    word=values[0]
    coefs=asarray(values[1:],dtype='float32')#获取单词对应的嵌入向量，并将其转换为 float32 类型的 NumPy 数组。这些向量在 GloVe 文件中是以字符串形式存储的，因此需要转换为数值形式。
    embedding_index[word]=coefs
f.close()
print("load %s word vectors"%len(embedding_index))

#为训练数据集中的每个单词创建一个嵌入矩阵
#我们可以通过枚举Tokenizer.word_index中的所有唯一单词并从加载的GloVe嵌入中找到嵌入权重向量来实现这一点
'''
glove.txt=glove.6B.50d(维度）
'''
embedding_matrix=zeros((vocab_size,50))              #初始化嵌入式矩阵，矩阵的行数等于词汇表大小（vocab_size），列数（纬度）为50
for word,i in tokenize.word_index.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

#构造模型

model=Sequential()
#嵌入层，用来降维
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len, weights=[embedding_matrix], trainable=False  ))#嵌入权重不会被更新
#将单词索引转换为固定大小的密集向量。处理文本数据的第一步，旨在捕捉单词之间的语义关系。
model.add(Bidirectional(LSTM(units=64,return_sequences=True)))#返回每个时间步的隐藏状态，供下一个LSTM层使用
model.add(Dropout(0.3))#正则化技术，用于减少过拟合。它在每次训练迭代中随机丢弃一定比例的神经元输出。
model.add(Bidirectional(LSTM(units=64,return_sequences=False)))#只返回序列的最终输出
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))#输入大小和输出大小的平均值,激活函数，快速的训练速度并减少梯度消失的问题。
#model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))#激活函数，适用于二分类问题
model.compile( loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(0.0012),#学习率
                metrics='accuracy')
print(model.summary())

#进行训练
x_train,x_test,y_train,y_test=train_test_split(text_data, targets, test_size=0.2,random_state=123,  shuffle=True)
cp=[ModelCheckpoint('model_bilstm.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True,mode='auto'),
    EarlyStopping(
    monitor='val_loss',  # 监控的指标
    patience=3,          # 在3个epoch内如果没有改善就停止
    verbose=1,           # 打印早期停止的日志
    restore_best_weights=True  # 恢复到最佳模型的权重
) ]
history=model.fit(text_data,targets,epochs=10,batch_size=32,shuffle=True, validation_split=0.2, validation_freq=1,  callbacks= cp)
#绘图

#测试
test_df=pd.read_csv('test.csv',encoding='utf-8')
#submit_example = pd.read_csv('submit_example.csv')
#test_df=test_df.drop(['News Url','Image Url','Report Content'],axis=1)
print(test_df.shape)
print(test_df.head())
print(test_df.isnull().sum())
#test_df=test_df.dropna()
#test_df=test_df.sample(frac=1).reset_index(drop=True)  tt['Title']+tt['Ofiicial Account Name']+
print(test_df.head())
tt=pd.DataFrame(test_df.astype(str))
test_df['text']=tt['content']#
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

submission = pd.DataFrame({'id': test_df['id'], 'label': predictions})
submission.to_csv('submit_example_bilstm.csv', index=False)
