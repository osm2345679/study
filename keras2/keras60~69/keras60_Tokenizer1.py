# Token : 데이터 처리 단위. 한 글자일 수도 있고 한 단어일 수도 있고 한 문장일 수도 있고 명확한 정의가 없는듯

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


text = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer() # 파이썬 기초 - 인스턴스(객체) = 클래스(), 인스턴스 생성
# 엄밀하게 말하면 객체(=object)가 큰 범위고 인스턴스(instance)는 클래스의 구현된 것으로서 클래스와의 관계가 강조되는 하위 범위임

token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# 많이 나온 것(high frequency), 먼저 나온 것 순으로 수치화됨.

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x) # [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]

### 데이터가 수치화됐으나 여기서의 수치(숫자)는 순위나 척도를 의미하는 것이 아니고 수치 간 관계가 없기 때문에
### 이를 동차원의 수치가 아니라 (다차원의 이진값으로) 위치로 변환시키는 one-hot encoding을 활용함.

print(len(x), len(x[0]))    # 1 14

# one-hot encoding 3 ways

# 변환 전 다루기 쉬운 nparray로, shape는 (N,)으로 변환
x = np.array(x).reshape(-1)

#1. pandas
# 어차피 pandas는 통으로 import함(import pandas as pd)
# dtype=int하면 깔끔함
pd_x = pd.get_dummies(x, dtype=int)
print(pd_x)
print(pd_x.shape)   # (14, 9)

#2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# sparse_out=False 넣고 shape 2차원으로 줘야 함
encoder = OneHotEncoder(sparse_output=False)
sk_x = encoder.fit_transform(x.reshape(-1, 1))
print(sk_x)
print(sk_x.shape)   # (14, 9)

#3. keras
# from tensorflow.keras.utils import to_categorical
# 0부터 인덱싱이니 데이터에 0 없으면 첫 줄 자르기
k = to_categorical(x)
k = k[:, 1:]
print(k)
print(k.shape)  # (14, 9)