import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
import jieba
from DataProcessor import DataCleaner


def PrepareDatas(FilePath, Headers, IsTrain=True):
    LoadedDatas = pd.read_csv(FilePath, usecols=Headers)
    LoadedDatas = DataCleaner(LoadedDatas)

    # 中文分词
    LoadedDatas['CombinedText'] = LoadedDatas.apply(lambda row: ' '.join(jieba.lcut(
        row['Title'] + " " + row['News Content'] + " " + row['Report Content'])), axis=1)

    if IsTrain:
        Labels = LoadedDatas['label']
        return LoadedDatas['CombinedText'], Labels
    else:
        Ids = LoadedDatas['id']
        return LoadedDatas['CombinedText'], Ids

