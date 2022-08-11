import pandas as pd
import requests


def NotUnitcode(x):
    if x[:2] != 'U+':
        return np.nan
    else:
        return x
    
    
url = 'https://unicode.org/emoji/charts/full-emoji-list.html'


html = pd.read_html(url)
df = html[0]

col = []
for i in range(15):
    col.append(df.columns[i][2])


df.columns = col
df['Code'] = df['Code'].map(lambda x: x[:7]).copy()
df['U_Code'] = df['Code'].apply(NotUnitcode)
emoji = df[~df['U_Code'].isna()][['Code','Browser','CLDR Short Name']].reset_index(drop=True)
emoji.to_csv('emoji.csv',index = False)
