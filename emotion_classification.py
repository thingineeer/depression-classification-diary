import os
from itertools import combinations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from konlpy.tag import Mecab
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler

from emoji_similarlity import Emoji_similarlity


model = AutoModelForSequenceClassification.from_pretrained("model_output_KcElectra")
tokenizer = AutoTokenizer.from_pretrained("tokenizer_KcElectra")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mecab = Mecab()
TAG_LIST = ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM","SF","SE","SSO","SSC","SC","SY", "VV"]
mmscaler = MinMaxScaler()

def pass_model(token):
    with torch.no_grad():
        output = model(**token)

    logits = output[0].detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    return result


def pass_model_for_mask(masked_loader):
    T = 0.95
    model.eval()

    masked_outputs=[]
    with torch.no_grad():
        for data in masked_loader:

            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            data_output = torch.sigmoid(outputs["logits"]).cpu().detach().numpy().tolist()

            # print(data_output)
            masked_ids = []
            # masked_ids.extend(ids)
            masked_outputs.extend(torch.sigmoid(outputs["logits"]).cpu().detach().numpy().tolist())
        
    # print(masked_outputs)
    # print(masked_ids)
    print("--"*50)
    return masked_outputs


def tokenize_list(list_to_tokenize):
    tokenized_list = tokenizer(
                    list_to_tokenize,
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    add_special_tokens=False
                )
    return tokenized_list


# 이거 안쓸 것임 (deprecated)
def word_predict(tokenized_sent):
    token = tokenizer.convert_ids_to_tokens([int(word) for word in tokenized_sent["input_ids"][0]][1:-1])
    pos = mecab.pos(''.join(token))

    emoji_sim = Emoji_similarlity()

    for pos_info in pos:
        if pos_info[1] not in TAG_LIST:
        
            token_word = tokenizer(
                pos_info[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )

            token_word.to(device)

            result = pass_model(token_word)
            if result == 0:
                print(f"악성 : {pos_info[0]}")
                #print(emoji_sim.cosine_ver(pos_info[0]))
        else:
            continue


class myDataset(torch.utils.data.Dataset):
    def __init__(self, encoding):
        self.encoding = encoding
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encoding.items()}
        return item

    def __len__(self):
        return len(self.encoding['input_ids'])


def word_predict_with_mask_without_mecab(tokenized_sent):

    token = [word for word in tokenized_sent["input_ids"][0]][1:-1]
    token_count = dict(zip(token, [0]*len(token)))

    mask_ids_combination = []
    [mask_ids_combination.extend(combinations(token,i)) for i in range(1, len(token)+1)]
    
    masked_token = {}
    masked_sentence = []
    where_mask = []
    for mask_ids in mask_ids_combination:
        where_mask.append(mask_ids)

        masked_token[mask_ids] = []

        for word in tokenized_sent["input_ids"][0]:
            if word in mask_ids:
                masked_token[mask_ids].append(torch.tensor(4).to(device))
            else:
                masked_token[mask_ids].append(word)

        masked_sentence.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(masked_token[mask_ids])))

    masked_dataset = tokenize_list(masked_sentence)
    
    mask_dataset = myDataset(masked_dataset)


    masked_params = {'batch_size': 8,
                    'shuffle': False,
                    'num_workers': 0
                    }

    masked_loader = DataLoader(mask_dataset, **masked_params)

    masked_outputs = pass_model_for_mask(masked_loader)
    print(f"masked_outputs : {masked_outputs}")
    T = 0.85
    for temp_output, temp_mask in zip(masked_outputs, where_mask):
        #print(temp_output, temp_mask, tokenizer.convert_ids_to_tokens(temp_mask))
        for word in temp_mask:
            if(temp_output[1] > T):
                token_count[word] += 1 
            
   
    scaled_count_values =  mmscaler.fit_transform(np.array(list(token_count.values())).reshape(-1,1))

    print(scaled_count_values)

    MMT = 0.8
    token_to_mask = []
    [token_to_mask.append(list(token_count.keys())[idx]) for idx in range(len(scaled_count_values)) if scaled_count_values[idx] >= MMT]
    # # # min-max scale 해서 0.9 이상을 마스킹
    
    print(tokenizer.convert_ids_to_tokens(token_count.keys()))
    print(tokenizer.convert_ids_to_tokens(token_to_mask))



def word_predict_with_mask(tokenized_sent):
    token = tokenizer.convert_ids_to_tokens([int(word) for word in tokenized_sent["input_ids"][0]][1:-1])
    pos = [mecab.pos(pos_of_token) for pos_of_token in token]

    ids_to_check = []
    new_pos = []
    for pos_info in pos:
        if len(pos_info) > 1:
            for pos_info_2 in pos_info:
                new_pos.append([pos_info_2])
        else:
            new_pos.append(pos_info)

    pos_without_tag_list = []
    [pos_without_tag_list.append(pos_info[0][0]) for pos_info in new_pos if pos_info[0][1] not in TAG_LIST]

    for token_item in token:
        for huh in pos_without_tag_list:
            if token_item == huh or token_item == "##"+huh:
                ids_to_check.append(tokenizer.convert_tokens_to_ids(token_item))

    # [ids_to_check.append(tokenized_sent["input_ids"][0][pos_idx+1]) for pos_idx in range(len(pos_without_tag_list))]

    ids_combination = []  # word to mask
    for i in range(1, len(ids_to_check)+1):
        ids_combination.extend(combinations(ids_to_check,i))

    print(ids_combination)
    for mask in ids_combination:
        origin = {}
        for idx in range(len(tokenized_sent["input_ids"][0])):
            if tokenized_sent["input_ids"][0][idx] in mask:
                origin[idx] = torch.tensor(tokenized_sent["input_ids"][0][idx])
                tokenized_sent["input_ids"][0][idx] = 4
            else:
                continue

        tokenized_sent.to(device)

        if pass_model(tokenized_sent) != 0:
            return tokenizer.convert_ids_to_tokens([int(word) for word in tokenized_sent["input_ids"][0]])
        else:
            for idx in origin.keys():
                tokenized_sent["input_ids"][0][idx] = origin[idx]
            continue

    # print(tokenizer.convert_ids_to_tokens([int(word) for word in tokenized_sent["input_ids"][0]]))
    return "경로를 벗어났습니다 이거 에러에요 확인하세요" # 
        

def sentence_predict(sent):
    model.eval()

    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )

    tokenized_sent.to(device)

    result = pass_model(tokenized_sent)

    # 악성댓글
    if result == 0:
        return word_predict_with_mask_without_mecab(tokenized_sent)
        # return word_predict_with_mask(tokenized_sent)

    #TODO: 합치기
    else:
        print(sent)
        return "정상댓글"

def split_sentence(sent):
    temp = sent.strip().split()
    sent_split = []
    for i in range(0,len(temp),4):
        sent_split.append(' '.join(temp[i:i+4]))

    return sent_split

def main():
    while True:
        sent = input("문장을 입력하세요: ")
        if sent == "0":
            break
    
        #TODO: 합치기
        for sent in split_sentence(sent):
            print(sentence_predict(sent))

        print("-"*50)   


if __name__ == "__main__":
    main()