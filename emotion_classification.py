import os

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


def pass_model(tokenzied_sent):
    with torch.no_grad():
        output = model(**tokenzied_sent)

    logits = output[0].detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    return result


def char_predict(tokenized_sent):

    token = tokenizer.convert_ids_to_tokens([int(char) for char in tokenized_sent["input_ids"][0]][1:-1])
    pos = mecab.pos(''.join(token))

    emoji_sim = Emoji_similarlity()

    for pos_info in pos:
        if pos_info[1] not in ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM","SF","SE","SSO","SSC","SC","SY"]:
            
            token_char = tokenizer(
                pos_info[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )

            tokenized_sent.to(device)

            result = pass_model(token_char)
            if result == 0:
                print(f"악성 : {pos_info[0]}")
                print(emoji_sim.cosine_ver(pos_info[0]))
        else:
            continue
            

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
        char_predict(tokenized_sent)

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