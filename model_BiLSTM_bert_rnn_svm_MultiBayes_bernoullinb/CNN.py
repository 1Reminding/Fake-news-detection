#导入语句
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from keras.layers import (GRU,LSTM,
                          Embedding,
                          Dense,
                          Dropout,
                          Bidirectional)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from keras.utils import pad_sequences
from nltk.stem import PorterStemmer,WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import jieba
from wordcloud import WordCloud,STOPWORDS
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, GlobalMaxPool1D
from keras.layers import Input
from keras import regularizers
from keras.layers import Conv1D, MaxPooling1D

#模型和数据准备
embedding_dim = 100# 设置嵌入层维度
embedding_path = 'glove.6B.100d.txt'  # GloVe嵌入文件路径

# 从GloVe文件中加载嵌入向量
embedding_index = {}
with open(embedding_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# 加载训练数据
train_df = pd.read_csv("train.news.csv")
train_df = train_df.dropna()# 删除空值
train_df = shuffle(train_df)# 随机打乱数据

pd.set_option('display.max_rows', None)# 设置显示最大行数

#文本预处理
stem = PorterStemmer()
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'# 定义去除标点符号的字符串

def stop_words_list(filepath):# 定义停用词加载函数
    stop_words = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stop_words
stopwords = stop_words_list('stop_words.txt')
def cleaning(text):# 定义文本清洗函数
    cutwords = list(jieba.lcut_for_search(text))
    final_cutwords = ''
    for word in cutwords:
        if word not in stopwords and punc:
            final_cutwords += word + ' '
    return final_cutwords

# 对训练集进行预处理
train_df["Report Content"] = train_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["data"].apply(cleaning)
train_data = train_df["data"]
texts = train_data
targets = np.asarray(train_df["label"])

import matplotlib.pyplot as plt

# DataFrame中的文本列名为 'data'
# 计算每条文本的长度（词数）
lengths = train_df['data'].apply(lambda x: len(x.split()))
# 打印文本长度的基本统计信息
print(lengths.describe())
# 绘制文本长度的直方图
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=30, color='blue', edgecolor='k', alpha=0.7)
plt.title('Text Length Distribution')
plt.xlabel('Length of Text')
plt.ylabel('Number of Texts')
plt.show()

# 对测试集进行相同的预处理
test_df=pd.read_csv("test.feature.csv",encoding='utf-8')
test_df["Report Content"] = test_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(test_df.astype(str))
test_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
text_test = test_df["data"]

'''

'''
#CountVectorizer进行特征提取
cv = CountVectorizer(min_df=0.1, max_df=0.9, ngram_range=(1, 2))
# transformed train reviews
cv_train_reviews = cv.fit_transform(train_data)
# transformed test reviews
cv_test_reviews = cv.transform(text_test)
cv_train_reviews = cv_train_reviews.sorted_indices()
cv_test_reviews = cv_test_reviews.sorted_indices()

print('BOW_cv_train:', cv_train_reviews.shape)
print('BOW_cv_test:', cv_test_reviews.shape)

MAX_NB_WORDS = 28455
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
MAX_SEQUENCE_LENGTH = 500
text_data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post',truncating='post')
EMBEDDING_DIM = 100#设置了嵌入层（embedding layer）的维度

num_words = min(MAX_NB_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < MAX_NB_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))  # 添加与数据形状匹配的输入层
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))

# 添加卷积层和池化层
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))

# 添加双向LSTM层
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(GlobalMaxPool1D())

# 添加全连接层和输出层
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
#L2正则化
model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(text_data, targets, batch_size=16, epochs=8, validation_split=0.2)
test_sequences = tokenizer.texts_to_sequences(text_test)
text_data_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

pred = model.predict(text_data_test)
preds =pred
# 预测和阈值处理
preds = model.predict(text_data_test)
predictions = (preds > 0.4).astype(int)#判真阈值

submission = pd.DataFrame({'id': test_df['id'], 'label': predictions.flatten()})
submission.to_csv('submit_CNN.csv', index=False)


#max 30 0.7930  25 低 35