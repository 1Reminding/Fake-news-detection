import numpy as np
import pandas as pd
import jieba
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.utils import shuffle
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

train_df = pd.read_csv("train.news.csv")
train_df = train_df.dropna()
train_df=shuffle(train_df)

punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
def stop_words_list(filepath):
    stop_words = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stop_words
stopwords = stop_words_list('stop_words.txt')
def cleaning(text):
    cutwords = list(jieba.lcut_for_search(text))
    final_cutwords = ''
    for word in cutwords:
        if word not in stopwords and punc:
            final_cutwords += word + ' '
    return final_cutwords

Stopwords = [line.strip() for line in open('stop_words.txt', 'r', encoding='utf-8').readlines()]


train_df["Report Content"] = train_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["data"].apply(cleaning)
train_data = train_df["data"]
x_train = train_data
y_train = np.asarray(train_df["label"])


test_df=pd.read_csv("test.feature.csv",encoding='utf-8')
test_df["Report Content"] = test_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(test_df.astype(str))
test_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
x_test = test_df["data"]



def tfidf(x_train, x_test):
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, x_test


def countvec(x_train, x_test):
    transfer=CountVectorizer(min_df=1, ngram_range=(1,1),stop_words=Stopwords)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, x_test

def tokenizer(x_train,x_test):
    MAX_NB_WORDS = 28455
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train)
    sequences = tokenizer.texts_to_sequences(x_train)
    word_index = tokenizer.word_index
    MAX_SEQUENCE_LENGTH = 300
    text_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    test_sequences = tokenizer.texts_to_sequences(x_test)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return text_data,test_data

#x_train, x_test = tfidf(x_train, x_test)
#x_train, x_test = countvec(x_train, x_test)
x_train,x_test = tokenizer(x_train,x_test)

estimator = SVC(kernel='rbf')
estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_test)
preds = y_predict
for i in range(len(preds)):
    if preds[i] > 0.5:
        preds[i] = 1
    else:
        preds[i] = 0


predictions =[]
for i in preds:
    predictions.append(i)
print(len(predictions))


submission = pd.DataFrame({'id': test_df['id'],'label':predictions})
submission.to_csv('submit_SVM_tokenizer.csv',index=False)






