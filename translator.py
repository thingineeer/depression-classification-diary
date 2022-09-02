from http import client
import os
import sys
import json
import urllib.request

with open('SECRETS.json') as r:
    secrets = json.load(r)
    client_id = secrets['client_id']
    client_secret = secrets['client_secret']

URL = "https://openapi.naver.com/v1/papago/n2mt"
client_id = client_id
client_secret = client_secret


def translate(source, target, text):

    encText = urllib.parse.quote(text)
    data = f"source={source}&target={target}&text=" + encText
    request = urllib.request.Request(URL)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    try:
        if(rescode==200):
            response_body = response.read()
            decode = json.loads(response_body.decode('utf-8'))
            result = decode['message']['result']['translatedText']
            return result
    
    except:
        raise Exception('Error Code:' + rescode)


def main():
    with open('./Dataset/emoji.csv', 'r') as f:
        for line in f:
            line = line.strip()
            source, target, text = line.split(',')
            translate(source, target, text)

    translate('en', 'ko', '')


if __name__ == "__main__":
    main()