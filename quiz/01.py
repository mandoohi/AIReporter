from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import os
import pandas as pd
import tensorflow as tf
import numpy as np

context = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

url = 'https://news.naver.com/main/clusterArticles.naver?id=c_202205041100_00001039&mode=LSD&mid=shm&sid1=100&oid=003&aid=0011167097'
request = Request(url, headers=headers)
response = urlopen(request, context=context)
html = response.read()

soup = BeautifulSoup(html, 'html.parser')
result = soup.find_all('a', {'class', 'nclicks(cls_pol.clsart1)'})
titles = []

for r in result:
    if r.text != '\n\n' and  r.text != '\n\n동영상기사\n':
        titles.append(r.text)

print(titles)

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
x = np.array([[]])
y = []
for a in range(1,len(sequence)+1):
    x[a:, 0] = sequence[:a]
    y[a] = sequence[a:a+1]
print(x)
print(y)




