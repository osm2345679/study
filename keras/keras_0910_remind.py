from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)
print(pd.DataFrame(y).isna())
print(np.unique(y, return_counts=True))

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측