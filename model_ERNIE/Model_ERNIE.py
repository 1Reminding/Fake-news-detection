import pandas as pd
import Reader
import DataProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import wandb
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import BertConfig, BertModel, BertTokenizer

# 预训练模型导入
TrainDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Config = BertConfig.from_json_file('ERNIE/bert_config.json')
Model = BertModel.from_pretrained('ERNIE/pytorch_model.bin', config=Config).to(TrainDevice)
Tokenizer = BertTokenizer.from_pretrained('ERNIE/vocab.txt')
print(f"Using device: {TrainDevice}")


# 模型定义
class AttentionLayer(nn.Module):
    def __init__(self, InputDim):
        super(AttentionLayer, self).__init__()
        self.AttentionWeights = nn.Linear(InputDim, 1)

    def forward(self, Inputs):
        Weights = F.softmax(self.AttentionWeights(Inputs), dim=1)
        Outputs = torch.sum(Weights * Inputs, dim=1)
        return Outputs


class FocalLoss(nn.Module):
    def __init__(self, Alpha=0.25, Gamma=2.0):
        super(FocalLoss, self).__init__()
        self.Alpha = Alpha
        self.Gamma = Gamma

    def forward(self, Inputs, Targets):
        BceLoss = F.binary_cross_entropy_with_logits(Inputs, Targets, reduction='none')
        Pt = torch.exp(-BceLoss)
        FLoss = self.Alpha * (1 - Pt) ** self.Gamma * BceLoss
        return FLoss.mean()


class Classifier(nn.Module):
    def __init__(self, DropoutRate=0.75):
        super(Classifier, self).__init__()
        self.AttnLayer = AttentionLayer(768)
        self.Fc1 = nn.Linear(768, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.Dropout1 = nn.Dropout(DropoutRate)
        self.Fc2 = nn.Linear(256, 128)
        self.BN2 = nn.BatchNorm1d(128)
        self.Dropout2 = nn.Dropout(DropoutRate)
        self.Fc3 = nn.Linear(128, 1)
        self.BN3 = nn.BatchNorm1d(1)

    def forward(self, Embeddings):
        AttnEmbeddings = self.AttnLayer(Embeddings)
        X = F.relu(self.BN1(self.Fc1(AttnEmbeddings)))
        X = self.Dropout1(X)
        X = F.relu(self.BN2(self.Fc2(X)))
        X = self.Dropout2(X)
        X = self.BN3(self.Fc3(X))
        return X.squeeze()


# 数据准备
def PrepareDatas(FilePath, Headers, IsTrain=True):
    LoadedDatas = Reader.CsvLoader(FilePath, Headers)
    LoadedDatas = DataProcessor.DataCleaner(LoadedDatas)
    CombinedText = LoadedDatas['Title'] + " " + LoadedDatas['News Content'] + " " + LoadedDatas['Report Content']
    if IsTrain:
        Labels = LoadedDatas['label']
        return CombinedText, Labels
    else:
        Ids = LoadedDatas['id']
        return CombinedText, Ids


def SlidingWindow(Text, TokenizerInUse, MaxLen=512, Stride=256):
    Tokens = TokenizerInUse(Text, add_special_tokens=True, return_tensors='pt')
    TokenTrunks = [Tokens.input_ids[:, i:i + MaxLen] for i in range(0, Tokens.input_ids.size(1), Stride)]
    Embeddings = []
    for Chunk in TokenTrunks:
        Chunk = Chunk.to(TrainDevice)
        AttentionMask = torch.ones(Chunk.shape, dtype=torch.long, device=TrainDevice)
        with torch.no_grad():
            Output = Model(Chunk, attention_mask=AttentionMask)
            Embeddings.append(Output.last_hidden_state.mean(dim=1))
    return torch.mean(torch.stack(Embeddings), dim=0)


def TextTokenizer(texts):
    return [Tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt") for text in
            texts]


def TrainModel(TrainPath, Epochs, LearnRate):
   # wandb.init(
    #    project="Fake News Detect",
     #   config=
      #  {
       #     "LearningRate": LearnRate,
        #    "Architecture": "ERNIE with Classifier",
         ##  "Epochs": Epochs
        #})
    Texts, Labels = PrepareDatas(TrainPath, ['Title', 'News Content', 'Report Content', 'label'], IsTrain=True)
    TrainTexts, ValidateTexts, TrainLabels, ValidateLabels = train_test_split(
        Texts, Labels, test_size=0.2, random_state=42, stratify=Labels)
    TrainTexts = TrainTexts.tolist()
    ValidateTexts = ValidateTexts.tolist()
    TrainDataSet = TensorDataset(torch.arange(len(TrainTexts)), torch.tensor(TrainLabels.values,
                                                                             dtype=torch.float))
    ValidateDataSet = TensorDataset(torch.arange(len(ValidateTexts)), torch.tensor(ValidateLabels.values,
                                                                                   dtype=torch.float))
    TrainDataLoader = DataLoader(TrainDataSet, batch_size=64, shuffle=True)
    ValidateDataLoader = DataLoader(ValidateDataSet, batch_size=64)
    TrainingModel = Classifier(DropoutRate=0.5).to(TrainDevice)
    Criterion = FocalLoss()
    Optimizer = optim.Adam(TrainingModel.parameters(), lr=LearnRate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(Optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    for Epoch in tqdm(range(Epochs), desc="Training Epochs"):
        TrainingModel.train()
        TotalTrainLoss = 0
        for Indices, Labels in TrainDataLoader:
            Texts = [TrainTexts[i] for i in Indices.numpy()]
            Labels = Labels.to(TrainDevice)
            Embeddings = [SlidingWindow(Text, Tokenizer, MaxLen=512) for Text in
                          Texts]
            Embeddings = torch.stack(Embeddings)
            Optimizer.zero_grad()
            Outputs = TrainingModel(Embeddings)
            CurTrainLoss = Criterion(Outputs, Labels)
            TotalTrainLoss += CurTrainLoss.item()
            CurTrainLoss.backward()
            Optimizer.step()
        AvgTrainLoss = TotalTrainLoss / len(TrainDataLoader)
        TrainingModel.eval()
        TotalValidateLoss = 0
        ValidatePredicts, ValidateLabels = [], []
        for Indices, Labels in ValidateDataLoader:
            Texts = [ValidateTexts[i] for i in Indices.numpy()]
            Labels = Labels.to(TrainDevice)
            Embeddings = [SlidingWindow(text, Tokenizer, MaxLen=512) for text in
                          Texts]
            Embeddings = torch.stack(Embeddings)
            with torch.no_grad():
                Outputs = TrainingModel(Embeddings)
                ValidateLoss = Criterion(Outputs, Labels)
                TotalValidateLoss += ValidateLoss.item()
                ValidatePredicts.append(Outputs.squeeze().cpu().numpy())
                ValidateLabels.append(Labels.cpu().numpy())
        AvgValidateLoss = TotalValidateLoss / len(ValidateDataLoader)
        ValidatePredicts = np.concatenate(ValidatePredicts)
        ValidateLabels = np.concatenate(ValidateLabels)
        AUC = roc_auc_score(ValidateLabels, ValidatePredicts)
     #   wandb.log({"Epoch": Epoch, "Train Loss": AvgTrainLoss, "Validate Loss": AvgValidateLoss, "Validate AUC": AUC})
        torch.save(TrainingModel.state_dict(),
                   f'TrainedModel/ERNIE/LearnRate{LearnRate}_Epoch{Epoch + 1}.pth')
        scheduler.step(AvgValidateLoss)
  #  wandb.finish()


def EvaluateModel(LearnRate, Epoch, PredictThreshold=0.5):
    ModelFile = f'TrainedModel/ERNIE/LearnRate{LearnRate}_Epoch{Epoch}.pth'
    TestFile = 'DataForTrain/Test.csv'
    EvalModel = Classifier()
    EvalModel.load_state_dict(torch.load(ModelFile))
    EvalModel.to(TrainDevice)
    EvalModel.eval()
    Texts, Ids = PrepareDatas(TestFile, ['id', 'Title', 'News Content', 'Report Content'], IsTrain=False)
    Texts = Texts.tolist()
    Predictions = []
    for Text in tqdm(Texts, desc="Evaluating"):
        with torch.no_grad():
            Embedding = SlidingWindow(Text, Tokenizer, MaxLen=512)
            Embedding = Embedding.unsqueeze(0).to(TrainDevice)
            Output = EvalModel(Embedding)
            Prediction = torch.sigmoid(Output).item()
            Predictions.append(1 if Prediction >= PredictThreshold else 0)
    ResultPath = f'PredictedLabel/ERNIE/Predictions{PredictThreshold}_LearnRate{LearnRate}_Epoch{Epoch}.csv'
    results = pd.DataFrame({
        "id": Ids,
        "label": Predictions
    })
    results.to_csv(ResultPath, index=False)
    return results
