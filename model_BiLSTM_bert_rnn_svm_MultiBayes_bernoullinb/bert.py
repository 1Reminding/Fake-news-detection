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


train_df = pd.read_csv("train.news.csv")
train_df = train_df.dropna()
train_df = shuffle(train_df)

stem = PorterStemmer()
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


train_df["Report Content"] = train_df["Report Content"].apply(lambda x:x.split("##"))
columns = ['Title', 'Report Content', 'label', 'Ofiicial Account Name']
t = pd.DataFrame(train_df.astype(str))
train_df['Title'] = t['Title'].apply(cleaning)
train_df['Report Content'] = t['Report Content'].apply(cleaning)
train_df['Ofiicial Account Name'] = t['Ofiicial Account Name']
train_df = train_df[columns]
data = train_df
print(data.head())
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(123)

model_name = "bert-base-chinese"
max_length= 512

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

data = data[data['Title'].notna()]
data = data[data['Ofiicial Account Name'].notna()]
data = data[data['Report Content'].notna()]


def prepare_data(df, test_size=0.2, include_title=True, include_author=True):
    texts = []
    labels = []

    for i in range(len(df)):
        text = df['Report Content'].iloc[i]
        label = df['label'].iloc[i]

        if include_title:
            text = df['Title'].iloc[i] + " - " + text
        if include_author:
            text = df['Ofiicial Account Name'].iloc[i] + " - " + text

        if text and label in [0, 1]:
            texts.append(text)
            labels.append(label)

    return train_test_split(texts, labels, test_size=test_size)


train_texts, valid_texts, train_labels, valid_labels = prepare_data(data)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]], dtype=torch.long)  # 强制转换为 torch.long 类型
        return item

    def __len__(self):
        return len(self.labels)


# convert tokenize data into torch dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

from sklearn.metrics import accuracy_score

def computer_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, }

training_args = TrainingArguments(
    output_dir='/results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=10,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    logging_dir='/results',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,               # log & save weights each logging_steps
    save_steps=200,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=computer_metrics,
)

trainer.train()
model.save_pretrained('./cache/model_bert1')
tokenizer.save_pretrained('./cache/tokenizer1')
def get_prediction(text, convert_to_label=False):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

test_df = pd.read_csv("test.feature.csv")
# make a copy of the testing set
new_df = test_df.copy()
# add a new column that contains the author, title and article content
new_df["Report Content"] = new_df["Report Content"].apply(lambda x:x.split("##"))
t = pd.DataFrame(train_df.astype(str))
new_df['Title'] = t['Title'].apply(cleaning)
new_df['Report Content'] = t['Report Content'].apply(cleaning)
new_df['Ofiicial Account Name'] = t['Ofiicial Account Name']
new_df = new_df[columns]
new_df["new_text"] = new_df["Ofiicial Account Name"].astype(str) + " " + new_df["Title"].astype(str) + " " + new_df["Report Content"].astype(str)
new_df["label"] = new_df["new_text"].apply(get_prediction)
# make the submission file
final_df = new_df[["id", "label"]]
final_df.to_csv("submit_Bert1.csv", index=False)