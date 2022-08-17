# Switch negative word

부적절한 단어를 댓글 속에서 감지하고 상응하는 이모지로 변경합니다.

<hr>

🤟🏻MEMBER🤟🏻

|  이사빈  |  정찬영  |  이명진  |  이성진  |
|--------|--------|--------|--------|
| <img src='https://avatars.githubusercontent.com/u/50198431?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/57498298?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/88179341?v=4' height=80 width=80></img> | <img src='https://avatars.githubusercontent.com/u/100620566?v=4' height=80 width=80></img> |
| [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/sabin5105) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/chanyoung1998) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/thingineeer) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/iamziny) |

<hr>

## 😌CONTENT😌

우리는 바람직한 인터넷 문화를 만들고자 합니다.

요즘 거대한 커뮤니티가 여기저기 있고 각자 대화를 나누지만, 댓글들을 보면 모두 클린하다고 볼 수는 없습니다.

그래서 우리는 댓글 속 비적절한 단어를 적절하게 바꿀 필요가 있다고 생각했습니다.
단순히 부정적인 댓글을 지우거나 ** 처리하는 것이 아니라 해당 단어를 상응하는 이모지로 변경함으로써 사용자에게 사소한 재미를 제공하고자 합니다.

<hr>

## 🚃Initial model🚃
| Classification | KcElectra | KoBERT | RoBERTa-base | RoBERTa-large |
| --- | --- | --- | --- | --- |
| Validation Accuracy | 0.88680 | 0.85721 | 0.83421 | 0.86994 |
| Validation Loss | 1.00431 | 1.23237 | 1.30012 | 1.16179 |
| Training Loss | 0.09908 | 0.03761 | 0.0039 | 0.06255 |
| Epoch | 10 | 40 | 20 | 20 |
| Batch-size | 8 | 32 | 16 | 32 |
| learning rate  | 2e-06 | 5e-5 | 2e-06 | 5e-6 |
| pretrained-model | beomi/KcELECTRA-base | skt/kobert-base-v1 | xlm-roberta-base | klue/roberta-large |

<hr>

## 💿DATASET💿

* Smilegate-AI / https://github.com/smilegate-ai/korean_unsmile_dataset

* naver 뉴스 기사 크롤링

## 🛠Used API🛠

* naver papago
* google translator

## 👨🏻‍💻What we've done👨🏻‍💻
- [x] 모델 선정 -> Fine tuning
- [x] pseudo labeling
- [x] masking the cursing words to predict which word is bad
- [x] binary classification for each comments
- [x] multi classification for each comments
- [x] calculation speed inhanced -> put batchs, split the sentences /.....
- [x] similarity(early deprecated😂)
- [x] pos tagging(also early deprecated...😂)
- [x] scaling

## TODO
- [ ] substitute cursing word with emoji
- [ ] increasing the model accuracy with hyperparameter tuning(Wandb)
- [ ] add more data with pseudo labeling
- [ ] launch server

## 🙋🏻HOW TO 🙋🏻

## 🔥REFERENCE🔥
