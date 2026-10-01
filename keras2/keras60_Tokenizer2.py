# 60-1 카피
# Token : 데이터 처리 단위. 한 글자일 수도 있고 한 단어일 수도 있고 한 문장일 수도 있고 명확한 정의가 없는듯

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다.'

token = Tokenizer() # 파이썬 기초 - 인스턴스(객체) = 클래스(), 인스턴스 생성
# 엄밀하게 말하면 객체(=object)가 큰 범위고 인스턴스(instance)는 클래스의 구현된 것으로서 클래스와의 관계가 강조되는 하위 범위임

token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, '개똥이는': 11, '기관사를': 12, '좋아한다': 13, '말똥이는': 14, '길동이는': 15, '더': 16}

x = token.texts_to_sequences([text1, text2])
print(x)    # [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

# one-hot encoding
x = np.concatenate(x)
print(x)    # [ 5  6  2  2  3  3  7  8  9  1  1  1  1 10 11 12 13 14  4 15  1  1 16  4]
print(x.shape)  # (24,)

encoder = OneHotEncoder(sparse_output=False)
x = encoder.fit_transform(x.reshape(-1,1))
print(x)
print(x.shape)  # (24, 16)