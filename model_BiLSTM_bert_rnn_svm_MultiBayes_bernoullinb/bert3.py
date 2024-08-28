import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import tqdm
import random
import numpy as np
from sklearn.metrics import matthews_corrcoef


# 设置种子
seed = 985
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else print('cuda不可用')

#加载训练和测试数据
train_df = pd.read_csv('train1.csv')
train_text= train_df['content']
train_label = train_df['label'].tolist()

test_df = pd.read_csv('test.feature.csv')
test_text= test_df['content'].apply(lambda x: ' '.join(map(str, x)), axis=1).tolist()


# 加载BERT模型和Tokenizer
model_name = 'bert-base-chinese'
model_path = r'C:\Users\元\Desktop\py\bert\bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

#数据预处理
def preprocess_texts(text, labels=None, max_length=80):
    input_ids = []
    attention_masks = []

    for text in text:
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if labels is not None:
        labels = torch.tensor(labels)
        return TensorDataset(input_ids, attention_masks, labels)
    else:
        return TensorDataset(input_ids, attention_masks)

#划分训练集和验证集
train_text, val_text, train_label, val_label= train_test_split(train_text, train_label, test_size=0.25, random_state=42)


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.dropout = torch.nn.Dropout(0.2)  # 可以调整dropout的比例

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)  # 在前向传播中应用 dropout

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



#创建DataLoader
train_dataset = preprocess_texts(train_text, train_label)
val_dataset = preprocess_texts(val_text, val_label)
test_dataset = preprocess_texts(test_text)
batch_size = 17
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
from transformers import get_linear_schedule_with_warmup


epochs = 4

from transformers import get_linear_schedule_with_warmup

# 在optimizer后面添加学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)



for epoch in range(epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)

    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()



        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {average_loss}")



#模型评估
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)





auc_score = roc_auc_score(all_labels, all_preds)



conf_matrix = confusion_matrix(all_labels, all_preds)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

print(f"Validation Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f'AUC Score: {auc_score}')

#测试测试集
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())

#结果保存
test_results = pd.DataFrame({'id': test_df['id'], 'label': all_preds})
test_results.to_csv('pre.csv', index=False)



