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

data_csv = 'hindi_train_val.csv'
stop_csv = 'stopwords.txt'
stop_words = 'stopwords'
df = pd.read_csv(io.BytesIO(uploaded[data_csv]))
df_test = pd.read_csv(io.BytesIO(uploaded['hindi_test.csv']))
stop_words = pd.read_csv(io.BytesIO(uploaded[stop_csv]))
stop_words.columns = ['stopwords']
stop_words = stop_words['stopwords'].tolist()
print(df)
print(stop_words)

def drop_stopwords(sent):
    #print(sent)
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

X = df.drop(['label'], axis = 1)
X = X.to_numpy()
y = df['label']
y = y.to_numpy()
X_temp = []

X_test = df_test.drop(['label'], axis = 1)
y_test = df_test['label']

X_temp_test = []

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

set_size=1
punctuations = ['!', '(', ')', '-', '[', ']', '{', '}', ';', ':', '\'', '\"', '\\', ',', '<', '>', '.', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~']
data_len = len(X)

for i in range(set_size-1, data_len):
    temp = np.array_str(X[i])
    curr_sent = ""
    for char in temp:
        if char not in punctuations:
            curr_sent +=char
    curr_sent = re.sub(r'\b\d+\b', ' ', curr_sent)
    curr_sent = re.sub(r'[^\u0900-\u097F|\d+|\w+]', ' ', curr_sent)
    curr_sent = drop_stopwords(curr_sent)
    X[i] = curr_sent

for i in range(0, len(X_test)):
    curr_sent = ""
    temp = np.array_str(X_test[i])
    for char in temp:
        if char not in punctuations:
            curr_sent = curr_sent + char
    #curr_sent = re.sub(r"[a-zA-Z0-9]", "", curr_sent)
    curr_sent = re.sub(r'[^\u0900-\u097F|\d+|\w+]', ' ', curr_sent)
    curr_sent = re.sub(r'\b\d+\b', ' ', curr_sent)
    curr_sent = drop_stopwords(curr_sent)
    X_test[i] = curr_sent
print(X)

temp = []
set_size=1
data_len = len(X)
Range = range(0, data_len)
for i in Range:
    temp.append(X[i][set_size-1])
X = temp
print(X)

temp_test = []
for i in range(0, len(X_test)):
    temp_test.append(X_test[i][0])
X_test = temp_test
print(X_test)

vectorizer = TfidfVectorizer(lowercase=False)
X_vector = vectorizer.fit_transform(X)
testing_size_data = 0.2
X_train, X_temp, y_train, y_temp = train_test_split(X_vector, y, test_size=testing_size_data, random_state=42)
X_test = X_temp
y_test = y_temp

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, f1_score

classifier = KNeighborsClassifier(n_neighbors=43)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Train Accurcy: {accuracy:.4f}')
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')
print(classification_report(y_test,y_pred))
Range_k = range(50,51)
scores = {}
scores_list = []

for k in Range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))

print("Test accuracy : ")
print(scores_list[0])

"""===============INTERSECTION FUNCTION======================"""

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
