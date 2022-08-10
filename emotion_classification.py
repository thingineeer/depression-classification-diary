import os

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model = AutoModelForSequenceClassification.from_pretrained("model_output")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    with torch.no_grad():
        output = model(**tokenized_sent)

    logits = output[0].detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    if result == 0:
        return "악성댓글"
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