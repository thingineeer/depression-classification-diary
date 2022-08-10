# Switch negative word

부적절한 단어를 댓글 속에서 감지하고 상응하는 이모지로 변경합니다.

<hr>

🤟🏻MEMBER🤟🏻

|  이사빈  |  정찬영  |  이명진  |  이성진  |
|--------|--------|--------|--------|
| <img src='https://avatars.githubusercontent.com/u/50198431?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/57498298?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/88179341?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/100620566?v=4' height=80 width=80></img> |
| [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/sabin5105) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/chanyoung1998) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/thingineeer) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/iamziny) |

<hr>

## CONTENT

우리는 바람직한 인터넷 문화를 만들고자 합니다.

요즘 거대한 커뮤니티가 여기저기 있고 각자 대화를 나누지만, 댓글들을 보면 모두 클린하다고 볼 수는 없습니다.

그래서 우리는 댓글 속 비적절한 단어를 적절하게 바꿀 필요가 있다고 생각했습니다.
단순히 부정적인 댓글을 지우거나 ** 처리하는 것이 아니라 해당 단어를 상응하는 이모지로 변경함으로써 사용자에게 사소한 재미를 제공하고자 합니다.

## Using model

"beomi/KcELECTRA-base" - HUGGINGFACE

<img width="500" alt="image" src="https://user-images.githubusercontent.com/50198431/183795775-a265070f-9b62-49c4-8341-d2650b4e67b2.png">

## DATASET

Smilegate-AI

<img width="400" alt="image" src="https://user-images.githubusercontent.com/50198431/183796088-59c55359-8307-48f2-b742-6d2fcd77162f.png">

https://github.com/smilegate-ai/korean_unsmile_dataset

```python
pd.read_csv("unsmile_train_v1.0.tsv", sep="\t").head()
```
<img width="1037" alt="image" src="https://user-images.githubusercontent.com/50198431/183796197-76e0ed1b-71f9-45a9-ba31-fe89e71ce15d.png">


## HOW TO 

## REFERENCE
