import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
# 加载训练数据
train_df = pd.read_csv("train.news.csv")
train_df = train_df.dropna()
train_df = shuffle(train_df)

# 定义停用词和清洗函数
def stop_words_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = [line.strip() for line in file.readlines()]
    return stop_words

stopwords = stop_words_list('stop_words.txt')

def cleaning(text):
    cutwords = list(jieba.lcut_for_search(text))
    final_cutwords = ''
    for word in cutwords:
        if word not in stopwords and punc:
            final_cutwords += word + ' '
    return final_cutwords

# 清洗训练数据
train_df["Report Content"] = train_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
train_df["data"] = train_df["data"].apply(cleaning)
x_train = train_df["data"]
y_train = np.asarray(train_df["label"])

# 加载和清洗测试数据
test_df = pd.read_csv("test.feature.csv", encoding='utf-8')
test_df["Report Content"] = test_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(test_df.astype(str))
test_df["data"] = t["Ofiicial Account Name"]+t["Title"]+t["Report Content"]
test_df["data"] = test_df["data"].apply(cleaning)
x_test = test_df["data"]

# 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(train_df['data'], train_df['label'], test_size=0.2, random_state=42)

# 特征提取
transfer = TfidfVectorizer(max_features=5000)
X_train = transfer.fit_transform(X_train)
X_val = transfer.transform(X_val)
X_test = transfer.transform(x_test)

# 训练朴素贝叶斯模型
alpha = 0.15
estimator = MultinomialNB(alpha=alpha)
estimator.fit(X_train, y_train)

# 在验证集上进行预测和评估
y_pred_val = estimator.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy)
print("Validation Recall:", recall)
print("Validation F1 Score:", f1)

# 对测试集进行预测
y_pred = estimator.predict(X_test)

# 准备提交文件
submission = pd.DataFrame({'id': test_df['id'], 'label': y_pred})
submission.to_csv('submit_Bayes.csv', index=False)
