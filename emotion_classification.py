import os
from itertools import combinations
import re

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from konlpy.tag import Mecab

from emoji_similarlity import Emoji_similarlity


model = AutoModelForSequenceClassification.from_pretrained("model_output_KcElectra")
tokenizer = AutoTokenizer.from_pretrained("tokenizer_KcElectra")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mecab = Mecab()
TAG_LIST = ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM","SF","SE","SSO","SSC","SC","SY"]


def pass_model(token):
    with torch.no_grad():
        output = model(**token)

    logits = output[0].detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    return result


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


def word_predict_with_mask(tokenized_sent):
    token = tokenizer.convert_ids_to_tokens([int(word) for word in tokenized_sent["input_ids"][0]][1:-1])
    pos = [mecab.pos(pos_of_token) for pos_of_token in token]

    ids_to_check = []
    [ids_to_check.append(tokenizer.convert_tokens_to_ids(pos_info[0][0])) for pos_info in pos if pos_info[0][1] not in TAG_LIST]

    ids_combination = []  # word to mask
    for i in range(1, len(ids_to_check)+1):
        ids_combination.extend(combinations(ids_to_check,i))
        
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

    return "경로를 벗어났습니다 이거 에러에요 확인하세요"
        

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
        return word_predict_with_mask(tokenized_sent)

    else:
        return "정상댓글"


def main():
    while True:
        sent = input("문장을 입력하세요: ")
        if sent == "0":
            break
        print(sentence_predict(sent))
        print("-"*50)   


if __name__ == "__main__":
    main()