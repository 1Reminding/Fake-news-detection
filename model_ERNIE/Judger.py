import pandas as pd
from sklearn.metrics import roc_auc_score


def CheckAUC(Threshold, Epochs, LearnRate, PredictThreshold=0.5):
    Predictions = pd.read_csv(f'Submit/Predictions{PredictThreshold}_LearnRate{LearnRate}_Threshold{Threshold}_Epoch{Epochs}.csv')
    Answers = pd.read_csv('Answer.csv')
    Predictions = Predictions.sort_values('id').reset_index(drop=True)
    Answers = Answers.sort_values('id').reset_index(drop=True)
    Auc = roc_auc_score(Answers['label'], Predictions['label'])
    print(f"AUC: {Auc}")
    return Auc


def OtherCheckAUC(Name):
    Predictions = pd.read_csv(f'PredictedLabel/{Name}')
    Answers = pd.read_csv('Answer.csv')
    Predictions = Predictions.sort_values('id').reset_index(drop=True)
    Answers = Answers.sort_values('id').reset_index(drop=True)
    Auc = roc_auc_score(Answers['label'], Predictions['label'])
    print(f"AUC: {Auc}")
    return Auc
