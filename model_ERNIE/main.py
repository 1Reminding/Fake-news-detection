import Judger
import Model_ERNIE
from Judger import CheckAUC
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from random import choice
import Model_NaiveBayes
import HeadOnly_Model_ERNIE

Model_ERNIE.TrainModel('DataForTrain/Train.csv', 10, 0.0005)
PredictThresholds = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
Epochs = [5, 6, 7, 8]
for PredictThreshold in PredictThresholds[::-1]:
    print(PredictThreshold)
    for Epoch in Epochs:
        print(Epoch)
        #Model_ERNIE.EvaluateModel(0.0005, Epoch, PredictThreshold)
        Judger.OtherCheckAUC(f'ERNIE/Predictions{PredictThreshold}_LearnRate{0.0005}_Epoch{Epoch}.csv')
# Model_ERNIE.EvaluateModel(0.0001, 1)

# 训练模型
# TrainPath = 'DataForTrain/Train.csv'  # 训练数据文件路径
# TestPath = 'DataForTrain/Test.csv'   # 测试数据文件路径
# ModelPath = 'TrainedModel/NaiveBayes/Bayes.joblib'  # 模型保存路径
# Model, Vectorizer = Model_NaiveBayes.TrainModel(TrainPath, ModelPath, 0.2)
# Model_NaiveBayes.EvaluateModel(ModelPath, TestPath, 'PredictedLabel/NaiveBayes/Bayes.csv')
# Judger.OtherCheckAUC('NaiveBayes/Bayes.csv')
