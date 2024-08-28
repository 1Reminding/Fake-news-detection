import re
from tqdm import tqdm


def DataCleaner(Datas):
    StopWords = set()
    with open('cn_stopwords.txt', 'r', encoding='UTF-8') as __:
        for _ in __:
            StopWords.add(_.strip())
    print('开始清理数据')
    for Index, Row in tqdm(Datas.iterrows(), total=Datas.shape[0]):
        Datas.at[Index, 'News Content'] = ContentProcessor(Datas.at[Index, 'News Content'], StopWords)
    return Datas


def CleanDataset(text):
    text = re.sub(r'#', '', text)
    text = re.sub(r'\&\w*;', '', text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    text = re.sub(r'\s\s+', '', text)
    text = re.sub(r'[ ]{2, }', ' ', text)
    text = re.sub(r'http(\S)+', '', text)
    text = re.sub(r'http ...', '', text)
    text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)
    text = re.sub(r'RT[ ]?@', '', text)
    text = re.sub(r'@[\S]+', '', text)
    text = re.sub(r'\b\w{1,4}\b', '', text)
    text = re.sub(r'&amp;?', 'and', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = ''.join(c for c in text if c <= '\uFFFF')
    text = text.strip()
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    text.replace('微信公众平台运营中心', '')
    text.replace('轻点两下取消赞 轻点两下取消在看', '')
    text.replace('此内容因违规无法查看', '')
    text.replace('由用户投诉并经平台审核', '')
    text.replace('账号已迁移', '')
    text.replace('该公众号已迁移', '')
    text.replace('公众号迁移说明', '')
    return text


def ContentProcessor(Text, StWords):
    Content = CleanDataset(Text)
    for StWord in StWords:
        Content.replace(StWord, '')
    return Content
