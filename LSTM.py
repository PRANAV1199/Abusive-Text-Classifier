# -*- coding: utf-8 -*-


! pip install cltk -q -U
! pip install demoji

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import re
import demoji
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()

data_csv = 'hindi_train_val1.csv'
stop_csv = 'stopwords.txt'
stop_words = 'stopwords'
df = pd.read_csv(io.BytesIO(uploaded[data_csv]))
df_test = pd.read_csv(io.BytesIO(uploaded[data_csv]))
stop_words = pd.read_csv(io.BytesIO(uploaded["stopwords.txt"]))
stop_words.columns = ['stopwords']
stop_words = stop_words['stopwords'].tolist()
print(df)
print(stop_words)

! pip install pytorch_lightning

from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import string

def drop_stopwords(sent):
    ret = ""
    words=[]
    start_index = 0
    for i in range(len(sent)):
      if sent[i] == ' ':
        words.append(sent[start_index:i])
        start_index = i + 1
    words.append(sent[start_index:])
    space=' '
    for word in words:
        if word not in stop_words:
            ret = ret + space + word
    return ret

set_size=1
X = df.drop(['label'], axis = set_size)
X = X.to_numpy()
y = df['label']

X_test = df_test.drop(['label'], axis = 1)
y_test = df_test['label']
y = y.to_numpy()
X_test = X_test.to_numpy()
X_temp = []
y_test = y_test.to_numpy()

punctuations = ['!', '(', ')', '-', '[', ']', '{', '}', ';', ':', '\'', '\"', '\\', ',', '<', '>', '.', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~']


X_temp_test = []

for i in range(set_size-1, len(X)):

    temp = np.array_str(X[i])
    curr_sent = ""
    for char in temp:
        if char not in punctuations:
            curr_sent = curr_sent + char
    #curr_sent = re.sub(r"[a-zA-Z0-9]", "", curr_sent)
    curr_sent = re.sub(r'\b\d+\b', ' ', curr_sent)
    curr_sent = re.sub(r'[^\u0900-\u097F|\d+|\w+]', ' ', curr_sent)
    curr_sent = drop_stopwords(curr_sent)
    X[i] = curr_sent

for i in range(set_size-1, len(X_test)):
    temp = np.array_str(X_test[i])
    curr_sent = ""
    for char in temp:
        if char not in punctuations:
            curr_sent = curr_sent + char
    #curr_sent = re.sub(r"[a-zA-Z0-9]", "", curr_sent)
    curr_sent = re.sub(r'\b\d+\b', ' ', curr_sent)
    curr_sent = re.sub(r'[^\u0900-\u097F|\d+|\w+]', ' ', curr_sent)
    curr_sent = drop_stopwords(curr_sent)
    X_test[i] = curr_sent

temp = []
set_size=1
for i in range(set_size-1, len(X)):
    temp.append(X[i][set_size-1])
X = temp
temp_test = []
print(X)
for i in range(set_size-1, len(X_test)):
    temp_test.append(X_test[i][set_size-1])
X_test = temp_test
print(X_test)

def preprocess_text(text):
    start_index = 0
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = []
    set_size=0
    str_len = len(text)
    for i in range(str_len):
      if text[i] == ' ':
        tokens.append(text[start_index:i+set_size])
        start_index = i + 1
    tokens.append(text[start_index:])
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token.lower() for token in tokens]

    text = ' '.join(tokens)
    return text

stop_csv = 'stopwords.txt'
data_csv = 'hindi_train_val.csv'
stop_words = 'stopwords'
stop_words = pd.read_csv(io.BytesIO(uploaded[stop_csv]))
stop_words.columns = ['stopwords']
stop_words = stop_words['stopwords'].tolist()
df = pd.read_csv(io.BytesIO(uploaded[data_csv]))
df_test = pd.read_csv(io.BytesIO(uploaded['hindi_test.csv']))
df['text'] = df['text'].apply(preprocess_text)
df_test['text'] = df_test['text'].apply(preprocess_text)

def convert_sent_to_list(sent):
    sent_list = []
    word_sent = []
    data_len = len(sent)
    Range_k = range(0, data_len)
    for i in Range_k:
        words = sent[i].strip().split(" ")
        curr_sent = []
        for word in words:
            if(word != ''):
                curr_sent.append(word)
                word_sent.append(word)
        sent_list.append(curr_sent)
    return sent_list

sent_list = convert_sent_to_list(df['text'])
sent_list_test = convert_sent_to_list(df_test['text'])

word_to_idx = {"<PAD>": 0, "<UNK>": 1}
SEQ_LEN = 100
set_size = 0

def preprocess(sent):
    for i in range(len(sent)):
        sent_data = len(sent[i])
        #Do padding in case of length not equal to maximum length
        while sent_data < SEQ_LEN:
            sent[i].append('<PAD>')
            sent_data = len(sent[i])

        if len(sent[i]) > SEQ_LEN:
            x_len = SEQ_LEN
            sent[i] = sent[i][:x_len]

    # So here we are giving an index to every word which is present in sentence
    for sentence in sent:
        for word in sentence:
            set_size=1
            if word not in word_to_idx:
                word_len = len(word_to_idx)
                word_to_idx[word] = word_len

    print(word_to_idx)
    idx_sent = [[word_to_idx.get(word, 1) for word in sentence] for sentence in sent]
    X_tensor = torch.tensor(idx_sent, dtype=torch.int).type(torch.LongTensor)
    return sent, X_tensor

sent_list, X_vocab = preprocess(sent_list)
set_size=1
def convert_sent_list_to_num_list(sent, vocab):
    Range_i = range(0, len(sent))
    for i in Range_i:
        for j in range(set_size-1, SEQ_LEN):
            sent[i][j] = word_to_idx.get(sent[i][j], set_size)

    return sent

sent_list = convert_sent_list_to_num_list(sent_list, X_vocab)
for i in range(len(sent_list_test)):
    while len(sent_list_test[i]) < SEQ_LEN:
        sent_list_test[i].append('<PAD>')

    if len(sent_list_test[i]) > SEQ_LEN:
        sent_list_test[i] = sent_list_test[i][:SEQ_LEN]

for i in range(0, len(sent_list_test)):
    for j in range(0, SEQ_LEN):
        sent_list_test[i][j] = word_to_idx.get(sent_list_test[i][j], 1)


print((sent_list[0]))
print((sent_list_test[0]))

class LSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout_rate, num_layers=1, bidirectional=False, learning_rate=0.01):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(embedding_dim,200, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        set_size=1
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, set_size)
        else:
            self.fc = nn.Linear(hidden_size, set_size)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))

        set_size=1
        lstm_out = self.dropout(lstm_out[:, set_size-2, :])
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        set_size=0
        outputs = self(inputs)
        set_size+=1
        loss = self.loss_fn(outputs, labels.unsqueeze(set_size).float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        set_size=1
        loss = self.loss_fn(outputs, labels.unsqueeze(set_size).float())
        self.log('val_loss', loss)
        predicted = torch.round(outputs)
        accuracy = (predicted == labels.unsqueeze(set_size).float()).sum().item() / labels.size(set_size-1)
        self.log('val_acc', accuracy)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        set_size=1
        predicted = torch.round(outputs)
        accuracy = (predicted == labels.unsqueeze(set_size).float()).sum().item() / labels.size(set_size-1)
        self.log('test_acc', accuracy)
        return accuracy

set_size=1
X = df.drop(['label'], axis = set_size)

y = df['label']
X = X.to_numpy()
y = y.to_numpy()

EMBEDDING_DIM = 512
HIDDEN_DIM    = 200
NUM_EPOCHS    = 10
BATCH_SIZE    = 32
dropout_rate = 0.5
num_layers = 2

y = torch.Tensor(y)
y_final_eval = torch.Tensor(y_test)
testing_size = 0.2
sent_list = torch.Tensor(sent_list).to(torch.int64)
sent_list_test = torch.Tensor(sent_list_test).to(torch.int64)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(sent_list, y, test_size=testing_size, random_state=42)
train_dataset = TensorDataset(X_train_lstm, y_train_lstm)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


val_dataset = TensorDataset(X_val_lstm, y_val_lstm)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataset = TensorDataset(sent_list_test, y_final_eval)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = LSTMModel(vocab_size=len(word_to_idx), embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_DIM, bidirectional=True, num_layers = num_layers, dropout_rate = dropout_rate)
early_stopping = EarlyStopping(monitor="train_loss", patience=3, mode="min")
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[early_stopping])
trainer.fit(model, train_dataloaders=train_loader)
trainer.test(dataloaders=val_loader)

#Test Accuracy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score

y_pred = []
y_true = []
model.eval()

for batch in val_loader:
    inputs, labels = batch
    outputs = model(inputs)
    predictions = (outputs.squeeze() > (5/10)).long()
    predictions =  predictions.tolist()
    y_pred.extend(predictions)
    y_true.extend(labels.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')
f1 = f1_score(y_true, y_pred)
print(f'F1 Score: {f1:.4f}')
print(classification_report(y_true,y_pred))

#TEST ACCURACY
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score


y_pred = []
y_true = []

for batch in test_loader:
    inputs, labels = batch
    outputs = model(inputs)
    predictions = (outputs.squeeze() > (5/10)).long()

    if(len(predictions.size()) == 0):
        continue
    y_pred.extend(predictions.tolist())
    y_true.extend(labels.tolist())

# calculate metrics
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')
f1 = f1_score(y_true, y_pred)
print(f'F1 Score: {f1:.4f}')

"""=======================INTERSECTION FUNCTION================================="""

def intersection_fun(X_1, X_2):
    common = 0
    Range_K = range(0, len(X_1))
    set_size = 0
    for i in Range_K:
        for j in range(set_size, len(X_2)):
            if(X_1[i] == X_2[j]):
                common =common + 1

    return common

df = pd.read_csv(io.BytesIO(uploaded[data_csv]))
df_test = pd.read_csv(io.BytesIO(uploaded['hindi_test.csv']))

X = df.drop(['label'], axis = 1)
y = df['label']
X = X.to_numpy()
X_temp = []
y = y.to_numpy()

set_size=1
X_test = df_test.drop(['label'], axis = set_size)
X_temp_test = []
y_test = df_test['label']


X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
###Calling Intersection Function
intersection_fun(X_test, X)
