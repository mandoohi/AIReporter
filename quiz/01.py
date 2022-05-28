from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import os
import pandas as pd
import tensorflow as tf
import numpy as np

context = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

url = 'https://news.naver.com/main/clusterArticles.naver?id=c_202205271700_00000001&mode=LSD&mid=shm&sid1=100&oid=277&aid=0005095529'
request = Request(url, headers=headers)
response = urlopen(request, context=context)
html = response.read()

soup = BeautifulSoup(html, 'html.parser')
result = soup.find_all('a', {'class', 'nclicks(cls_pol.clsart1)'})
titles = []

for r in result:
    if r.text != '\n\n' and  r.text != '\n\n동영상기사\n':
        titles.append(r.text)

if not os.path.exists('../data'):
    os.mkdir('../data')

data = pd.DataFrame({'title': titles})
data.to_csv('../data/titles.csv', encoding='utf-8')
data = pd.read_csv('../data/titles.csv')
titles = data['title'].values
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
print(tokenizer.word_index)
sequence = tokenizer.texts_to_sequences([titles[0]])[0]
x = []
y = []
print(titles[0])
print(sequence)
for a in range(1, len(sequence)):
    x.append(sequence[:a])
    y.append(sequence[a])
print('x = ', x)
print('y= ', y)
word_count = len(tokenizer.word_index)+1
max_len_y = 0
max_len = 0
for i in range(len(x)):
    max_len = max(max_len, len(x[i]))
for i in range(len(y)):
    max_len_y = max(max_len_y, int(y[i]))
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
print(pad_sequences)
categorical_data = tf.keras.utils.to_categorical(y, num_classes = max_len_y+1)
print(categorical_data)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max(sequence)+1,5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(max(sequence)+1),
    tf.keras.layers.Softmax()
])

predict = model.predict([sequence])

argmax = np.argmax(predict[0])
print(predict)

