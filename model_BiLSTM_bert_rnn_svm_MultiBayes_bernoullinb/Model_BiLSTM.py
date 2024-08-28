import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import wandb
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import functional as F
from Reader import CsvLoader
from DataProcessor import DataCleaner

EmbeddingDim = 128
HiddenDim = 256
OutputDim = 1
NLayers = 2
Dropout = 0.5


class FocalLoss(nn.Module):
    def __init__(self, Alpha=0.25, Gamma=2.0):
        super(FocalLoss, self).__init__()
        self.Alpha = Alpha
        self.Gamma = Gamma

    def forward(self, Inputs, Targets):
        BceLoss = F.binary_cross_entropy(Inputs, Targets, reduction='none')
        Pt = Inputs * Targets + (1 - Inputs) * (1 - Targets)
        FLoss = self.Alpha * (1 - Pt) ** self.Gamma * BceLoss
        return FLoss.mean()


# 双向LSTM模型定义
class BiLSTM(nn.Module):
    def __init__(self, VocabSize, EmbeddingDim, HiddenDim, OutputDim, NLayers, Dropout=0.5):
        super().__init__()
        self.Embedding = nn.Embedding(VocabSize, EmbeddingDim)
        self.RNN = nn.LSTM(EmbeddingDim, HiddenDim, num_layers=NLayers, bidirectional=True, dropout=Dropout)
        self.Fc = nn.Linear(HiddenDim * 2, OutputDim)
        self.Dropout = nn.Dropout(Dropout)

    def forward(self, Text):
        Text = Text.permute(1, 0)
        Embedded = self.Dropout(self.Embedding(Text))
        Output, (Hidden, Cell) = self.RNN(Embedded)
        Hidden = torch.cat((Hidden[-2, :, :], Hidden[-1, :, :]), dim=1)
        Hidden = self.Dropout(Hidden)
        Out = self.Fc(Hidden)
        return torch.sigmoid(Out).squeeze(-1)


# 数据集定义
class TextDataset(Dataset):
    def __init__(self, Texts, Labels):
        self.Texts = Texts
        self.Labels = Labels

    def __len__(self):
        return len(self.Texts)

    def __getitem__(self, Index):
        return self.Texts[Index], self.Labels[Index]


def Tokenizer(Texts, VocabSize=10000):
    Tokens = [jieba.lcut(Text) for Text in Texts]
    AllWords = [Word for SubList in Tokens for Word in SubList]
    WordCounts = Counter(AllWords)
    Vocab = [Word for Word, Count in WordCounts.most_common(VocabSize)]
    Word2Index = {Word: Index + 2 for Index, Word in enumerate(Vocab)}
    Word2Index["<UNK>"] = 1
    Word2Index["<PAD>"] = 0
    return Tokens, Word2Index


def TextEncoder(Tokens, Word2Index, MaxLen=100):
    EncodedTexts = []
    for Text in Tokens:
        EncodedText = [Word2Index.get(Word, 1) for Word in Text][:MaxLen]
        EncodedText += [0] * (MaxLen - len(EncodedText))
        EncodedTexts.append(EncodedText)
    return np.array(EncodedTexts)


def TrainModel(LearnRate=0.0005, Epochs=30):
    wandb.init(project="Fake News Detect", config={
        "LearningRate": LearnRate,
        "Architecture": "BiLSTM",
        "Dataset": "WeChat News",
        "Epochs": Epochs
    })
    Data = CsvLoader('DataForTrain/Train.csv', ['Title', 'News Content', 'label'])
    Data = DataCleaner(Data)
    Tokens, Word2Index = Tokenizer(Data['Title'])
    X = TextEncoder(Tokens, Word2Index)
    y = Data['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    TrainData = TextDataset(X_train, y_train)
    ValidateData = TextDataset(X_val, y_val)
    TrainLoader = DataLoader(TrainData, batch_size=32, shuffle=True)
    ValidateLoader = DataLoader(ValidateData, batch_size=32)
    # 模型参数
    VocabSize = len(Word2Index)

    BiLSTMModel = BiLSTM(VocabSize, EmbeddingDim, HiddenDim, OutputDim, NLayers, Dropout)
    Optimizer = optim.Adam(BiLSTMModel.parameters(), lr=LearnRate)
    Criterion = FocalLoss()
    TrainDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BiLSTMModel = BiLSTMModel.to(TrainDevice)
    Criterion = Criterion.to(TrainDevice)
    pd.to_pickle(Word2Index, 'TrainedModel/Word2Index.pkl')
    for Epoch in tqdm(range(Epochs), desc="Training Epochs"):
        BiLSTMModel.train()
        TrainLosses = []
        for Texts, Labels in TrainLoader:
            Texts, Labels = Texts.to(TrainDevice), Labels.to(TrainDevice)
            Optimizer.zero_grad()
            Predictions = BiLSTMModel(Texts)
            Loss = Criterion(Predictions, Labels.float())
            Loss.backward()
            Optimizer.step()
            TrainLosses.append(Loss.item())
        AvgTrainLoss = sum(TrainLosses) / len(TrainLosses)
        wandb.log({"Train Loss": AvgTrainLoss})
        BiLSTMModel.eval()
        AllPredictions = []
        AllLabels = []
        with torch.no_grad():
            ValidateLosses = []
            for Texts, Labels in ValidateLoader:
                Texts, Labels = Texts.to(TrainDevice), Labels.to(TrainDevice)
                Predictions = BiLSTMModel(Texts)
                Loss = Criterion(Predictions, Labels.float())
                ValidateLosses.append(Loss.item())
                AllPredictions.extend(Predictions.cpu().numpy())
                AllLabels.extend(Labels.cpu().numpy())
            AvgValidateLoss = sum(ValidateLosses) / len(ValidateLosses)
            AUC = roc_auc_score(AllLabels, AllPredictions)  # 计算 AUC
            wandb.log({"Validate Loss": AvgValidateLoss, "Validate Auc": AUC, "Epoch": Epoch})
            torch.save(BiLSTMModel.state_dict(), f'TrainedModel/BiLSTM_LearnRate{LearnRate}_Epoch{Epoch}.pth')

    wandb.finish()


def EvaluateModel(Epoch, LearnRate=0.0005, Threshold=0.5):
    Word2Index = pd.read_pickle('TrainedModel/Word2Index.pkl')
    VocabSize = len(Word2Index)
    BiLSTMModel = BiLSTM(VocabSize, EmbeddingDim, HiddenDim, OutputDim, NLayers, Dropout)
    BiLSTMModel.load_state_dict(torch.load(f'TrainedModel/BiLSTM_LearnRate{LearnRate}_Epoch{Epoch}.pth',
                                           map_location=torch.device('cpu')))
    TestData = CsvLoader('DataForTrain/Test.csv', ['id', 'Title', 'News Content'])
    TestData = DataCleaner(TestData)
    Tokens = [jieba.lcut(Text) for Text in TestData['Title']]
    XTest = TextEncoder(Tokens, Word2Index)
    TestDataset = TextDataset(XTest, np.zeros(len(XTest)))
    TestLoader = DataLoader(TestDataset, batch_size=32)
    BiLSTMModel.eval()
    predictions = []
    with torch.no_grad():
        for texts, _ in TestLoader:
            outputs = BiLSTMModel(texts)
            predictions.extend(outputs.numpy())
    TestData['label'] = (np.array(predictions) >= Threshold).astype(int)
    TestData[['id', 'label']].to_csv(f'PredictedLabel/BiLSTM_Predictions{Threshold}_'
                                     f'LearnRate{LearnRate}_Epoch{Epoch}.csv', index=False)
TrainModel(LearnRate=0.0005, Epochs=30)
EvaluateModel(Epoch=10, LearnRate=0.0005, Threshold=0.5)
