import pandas as pd
import numpy as np
import jieba
import os
import csv
import jieba.analyse
from sklearn.utils import shuffle
from nltk.stem import PorterStemmer
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import random
import wandb
wandb.login(key='737f2bbdefed89aeeea9e69073995c88b7da8336')

#数据预处理
#1.	加载和清洗数据
train_df = pd.read_csv("train1.csv")#使用 pandas 读取 train.news.csv 文件，该文件包含训练用的新闻数据。
train_df = train_df.dropna()#函数删除数据中的空值
train_df = shuffle(train_df)#打乱数据

stem = PorterStemmer()
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'

#2.	文本清洗
def stop_words_list(filepath):#从一个文件中读取并创建停用词列表
    stop_words = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stop_words
stopwords = stop_words_list('stop_words.txt')
def cleaning(text):#清洗给定的文本，移除其中的停用词和标点符号
    cutwords = list(jieba.lcut_for_search(text))  #将连续的文本分割成有意义的词语（tokens）
    final_cutwords = ''
    for word in cutwords:
        if word not in stopwords and punc:  # 去除标点punc（排空）
            final_cutwords += word + ' '
    return final_cutwords  #构建最终文本
'''
单纯地检查 punc 字符串是否非空，而不是检查词语是否包含标点。
如果停用词列表非常长，检查一个词是否在停用词列表中可能会有些慢。考虑将停用词列表转换为集合（set），因为在集合中查找元素通常比在列表中更快。
'''
#保证数据的结构是适合后续处理
#设置分割函数，分割 'Report Content' 列
train_df["content"] = train_df["content"].apply(lambda x:x.split("##"))
#定义了一个列名列表，这些列名将用于后续操作
#columns = ['Title', 'Report Content', 'label', 'Ofiicial Account Name']
columns = ['content', 'label']
#创建临时DataFrame并应用清洗函数
t = pd.DataFrame(train_df.astype(str))#所有元素都转换为字符串类型。这是为了确保后续操作中处理的是文本数据
#train_df['Title'] = t['Title'].apply(cleaning)#去除文本中的停用词和标点符号，以提高数据质量
train_df['content'] = t['content'].apply(cleaning)
#train_df['Ofiicial Account Name'] = t['Ofiicial Account Name']
#确保了DataFrame的结构符合预期，即包含清洗后的 'Title' 和 'Report Content'，以及原始的 'label' 和 'Ofiicial Account Name'
train_df = train_df[columns]
data = train_df
print(data.head())

#模型训练
def set_seed(seed: int):#提供一致的随机数生成，帮助确保实验结果的一致性和可重复性。数据的随机分配和模型初始化将保持一致
    random.seed(seed)
    np.random.seed(seed)#影响所有基于 NumPy进行随机数生成的操作，PyTorch 框架中所有随机数生成的行为， GPU 上运行的 PyTorch 操作的随机性，TensorFlow 框架中所有随机数生成的行为
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(985)#随意设置整数，不同的种子值会生成不同的随机数序列

model_name = "bert-base-chinese"#专门针对中文文本处理优化的BERT模型
max_length= 512#BERT模型有固定的最大序列长度限制

#初始化一个用于文本处理的分词器（tokenizer），将原始文本数据转换为模型可以理解的格式，即将文本分解为BERT模型可以处理的令牌（tokens）
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)#do_lower_case英文字符转换为小写

#进一步清洁和准备数据，确保用于模型训练和分析的数据集中没有缺失值。移除各列中含有空值的行
#data = data[data['Title'].notna()]
#data = data[data['Ofiicial Account Name'].notna()]
data = data[data['content'].notna()]
'''
.notna() 是 Pandas 提供的一个方法，用来检查数据帧中的元素是否不是 NaN
保证数据有效完整无缺失
'''

def prepare_data(df, test_size=0.2, include_title=True, include_author=True):#20%的数据用作测试集，其余的用作训练集
    texts = []
    labels = []

    for i in range(len(df)):
        text = df['content'].iloc[i]
        label = df['label'].iloc[i]

        #if include_title:
          #  text = df['Title'].iloc[i] + " - " + text
        #if include_author:
           # text = df['Ofiicial Account Name'].iloc[i] + " - " + text

        if text and label in [0, 1]:
            texts.append(text)
            labels.append(label)

    return train_test_split(texts, labels, test_size=test_size)
#准备数据以符合模型训练所需的格式，即分离特征（文本）和目标变量（标签）。
#数据集分割和文本编码
train_texts, valid_texts, train_labels, valid_labels = prepare_data(data)#调用前面定义的 prepare_data 函数，将 data DataFrame 分割为训练集和验证集
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)#用前面初始化的BERT分词器对训练集文本进行编码，过长被截断，过短填充，保证文本长度一致，BERT模型要求输入长度一致
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

#自定义的PyTorch数据集类，PyTorch中处理数据的标准形式，便于后续在模型中使用
class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):#构造函数，用于初始化数据集实例
        self.encodings = encodings#编码数据和标签
        self.labels = labels

    def __getitem__(self, idx):#访问单个数据项
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]], dtype=torch.long)  # 强制转换为 torch.long 类型
        return item

    def __len__(self):#返回数据集中样本的总数
        return len(self.labels)
#通过继承 torch.utils.data.Dataset，NewsGroupsDataset 类与PyTorch框架兼容，可以与PyTorch的数据加载器（DataLoader）一起使用，以高效地加载和批处理数据。

#创建训练集数据集
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
#创建验证集数据集
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
'''
确保了数据以适当的方式被封装和管理，以便于后续的模型训练和评估过程。为使用 PyTorch 进行模型训练和验证做好了准备,用于高效地加载数据、批处理和迭代。
'''
#6.	模型配置和训练.加载并初始化一个用于序列分类任务的BERT模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)#二分类
from sklearn.metrics import accuracy_score
#计算模型的准确率
def computer_metrics(pred):#计算并输出模型的性能指标
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, }

training_args = TrainingArguments(
    output_dir='/results',          # 模型训练过程中生成的所有输出（如模型检查点）将被保存在这个目录。
    num_train_epochs=5,              # 总训练轮次，即整个训练数据集将被遍历的次数。
    per_device_train_batch_size=10,  # 在训练过程中每个设备（如GPU）上的批处理大小。
    per_device_eval_batch_size=20,   # 在评估过程中每个设备上的批处理大小。
    warmup_steps=100,                # 学习率预热步数，这是训练初期逐渐增加学习率的步数。
    logging_dir='/results',            # 用于存储训练日志的目录。
    load_best_model_at_end=True,     # 训练完成后是否加载性能最佳的模型。
    logging_steps=200,               # 每隔多少步记录一次日志
    save_steps=200,                  # 每隔多少步保存一次模型。
    evaluation_strategy="steps",     # 评估策略，这里设置为在每个 logging_steps 后进行一次评估。
)
#transformers 库中的 Trainer 类
trainer = Trainer(#训练的模型，参数，用于训练的数据集，用于评估的数据集
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=computer_metrics,#用于计算评估指标的函数
)
#开始训练过程。训练完成后，模型的最佳版本（如果 load_best_model_at_end 为 True）将被保存。
trainer.train()
#保存模型和分词器
model.save_pretrained('./cache/model_bert1')#保存训练后的模型到指定目录。
tokenizer.save_pretrained('./cache/tokenizer1')#保存使用过的分词器到指定目录。
def get_prediction(text, convert_to_label=False):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())
test_df = pd.read_csv("test.feature.csv")
# 创建副本。保留原始数据不变的同时对数据副本进行修改或分析
new_df = test_df.copy()

# 创建了一个新的列 'new_text'，它通过将 'Ofiicial Account Name'、'Title' 和 'Report Content' 列的内容合并来形成每个样本的完整文本。各部分之间用空格分隔。
new_df["content"] = new_df["content"].apply(lambda x:x.split("##"))
#new_df['Title'] = new_df['Title'].apply(cleaning) # 修改这里
new_df['content'] = new_df['content'].apply(cleaning) # 修改这里
#new_df['Ofiicial Account Name'] = new_df['Ofiicial Account Name'] # 修改这里new_df["Ofiicial Account Name"].astype(str) + " " + new_df["Title"].astype(str) + " " +
new_df["new_text"] = new_df["content"].astype(str)

# 应用模型进行预测
new_df["label"] = new_df["new_text"].apply(get_prediction)

# make the submission file
final_df = new_df[["id", "label"]]
final_df.to_csv("submit_Bert1.csv", index=False)