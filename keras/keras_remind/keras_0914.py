from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
# print(x, y)
print(pd.DataFrame(x).isna())
print(pd.DataFrame(y).isna())

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측