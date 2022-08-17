List=[]

from bs4 import BeautifulSoup
import requests
import re
import sys
import pprint


url="http://news.naver.com/main/hotissue/read.nhn?mid=hot&sid1=100&cid=1079165&iid=2780573&oid=001&aid=0010079145&ptype=052"


oid=url.split("oid=")[1].split("&")[0]
aid=url.split("aid=")[1]
page=1    
header = {
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
    "referer":url,
    
} 
while True :
    c_url="https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news"+oid+"%2C"+aid+"&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page="+str(page)+"&refresh=false&sort=FAVORITE" 
# 파싱
    r=requests.get(c_url,headers=header)
    cont=BeautifulSoup(r.content,"html.parser")    
    total_comm=str(cont).split('comment":')[1].split(",")[0]
   
    match=re.findall('"contents":([^\*]*),"userIdNo"', str(cont))

    List.append(match)
# 한번에 댓글이 20개씩 보이기 때문에 한 페이지씩 몽땅 댓글을 긁어 옵니다.
    if int(total_comm) <= ((page) * 20):
        break
    else : 
        page+=1



def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList 
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList



##### 한국어만 보이게 합니다.

def only_korean(sentence_list):
    new_list = []
    for i in sentence_list:
        i = re.sub('\"','',i)
        new_list.append(' '.join(re.compile('[가-힣]+').findall(i)))
    return new_list
    
only_korean(flatten(List))

