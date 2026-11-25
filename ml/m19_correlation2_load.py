from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import joblib
import pickle

#1. 데이터

# load
path = './_save/'
# df = joblib.load(path + 'joblib/m19_correlation1.joblib')

with open(path + 'pickle/m19_correlation1.pickle', 'rb') as f:
    df = pickle.load(f)

# print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
# 0                  5.1               3.5                1.4               0.2       0
# 1                  4.9               3.0                1.4               0.2       0
# ..                 ...               ...                ...               ...     ...
# 149                5.9               3.0                5.1               1.8       2
# [150 rows x 5 columns]

print("========== 상관관계 히트맵 ==========")
print(df.corr())
# ========== 상관관계 히트맵 ==========
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
# Target                      0.782561         -0.426658           0.949035          0.956547  1.000000

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()


#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측