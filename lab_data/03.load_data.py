import pandas as pd

data = pd.read_csv('../data/titles.csv')
print(data)

titles = data['title'].values
print(titles)





