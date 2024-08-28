import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import json


def WebReader(Url):
    Response = requests.get(Url)
    Response.raise_for_status()
    Soup = BeautifulSoup(Response.text, 'html.parser')
    Text = Soup.get_text(separator=' ', strip=True)
    return Text


def CsvLoader(CsvPath, TargetDatas):
    return pd.read_csv(CsvPath, usecols=TargetDatas)


def GetNewsContent(DataHeaders, FilePath, SavePath):
    LoadedDatas = CsvLoader(FilePath, DataHeaders)
    LoadedDatas['News Content'] = None
    for Index, Row in tqdm(LoadedDatas.iterrows(), total=LoadedDatas.shape[0]):
        LoadedDatas.at[Index, 'News Content'] = WebReader(Row['News Url'])
    while any("验证 操作频繁，请稍后再试" in Content for Content in LoadedDatas['News Content']):
        for Index, Content in enumerate(LoadedDatas['News Content']):
            if "验证 操作频繁，请稍后再试" in Content:
                LoadedDatas.at[Index, 'News Content'] = WebReader(LoadedDatas.at[Index, 'News Url'])
    LoadedDatas.to_csv(SavePath, index=False)


def DownloadNewsContent():
    TrainDataHeaders = ['Official Account Name', 'Title', 'News Url', 'Report Content', 'label']
    TrainFP, TrainSP = 'DataForTrain/train.news.csv', 'DataForTrain/GatheredTrain.news.csv'
    print('Downloading training datas')
    GetNewsContent(TrainDataHeaders, TrainFP, TrainSP)
    ExamDataHeaders = ['id', 'Official Account Name', 'Title', 'News Url', 'Report Content']
    ExamFP, ExamSP = 'DataForTrain/test.feature.csv', 'DataForTrain/GatheredTest.feature.csv'
    print('Downloading exam datas')
    GetNewsContent(ExamDataHeaders, ExamFP, ExamSP)


def JsonLoader(JsonPath):
    with open(JsonPath, 'r') as File:
        Data = [json.loads(Line) for Line in File]
    return Data
