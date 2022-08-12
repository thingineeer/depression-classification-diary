import googletrans

translator = googletrans.Translator()

z = []
for i in emoji_dict.values():
    z.append(translator.translate(i, src='en', dest='ko').text) # 생각보다 오래걸림
    
 df = pd.DataFrame(list(emoji_dict.items()), columns=['Code', 'CLDR Short Name'])

df['Tran_Name'] = z
df.to_csv('Trans_Name.csv', index=False)

# pd.read_csv('Trans_Name.csv')
