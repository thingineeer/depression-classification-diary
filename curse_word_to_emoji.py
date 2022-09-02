import os
from itertools import combinations
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from konlpy.tag import Mecab
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler

emoji = pd.read_csv("emoji_category.csv")
#모델이랑 tokenizer 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained("multi_classification")
tokenizer = AutoTokenizer.from_pretrained("tokenizer_KcElectra")

def sentence_multi_classification(sentence):
    columns = ['악플/욕설', '인종/국적', '개인지칭', '연령', '남성', '여성/가족', '성소수자', '지역','기타 혐오', '종교']
    tokenized_sent = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    
    tokenized_sent.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(**tokenized_sent)
        
    outputs = output['logits'].detach().cpu().numpy()
    final_outputs = np.array(outputs) >=0.5
    
    return (pd.DataFrame(final_outputs,columns=columns))

def curse_word_to_emoji(word):
    columns = ['악플/욕설', '인종/국적', '개인지칭', '연령', '남성', '여성/가족', '성소수자', '지역','기타 혐오', '종교']
    random_emoji = []
    final_outputs = sentence_multi_classification(word)
    print(final_outputs)

    for col in columns:
        if final_outputs.loc[0,col] == True:
                random_emoji.extend(emoji.loc[emoji["category"] == col,"Browser"].tolist())
        
    
    if random_emoji == []:
        random_emoji.extend(emoji.loc[emoji["category"] == '기타',"Browser"].tolist())
    
    return random.sample(random_emoji,1)[0]
    
    

    
    
print(curse_word_to_emoji("병신"))



    
    
