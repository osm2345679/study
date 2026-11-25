from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import joblib
import pickle

#1. 데이터
datasets = load_iris()
# print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
# print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  5.1               3.5                1.4               0.2
# 1                  4.9               3.0                1.4               0.2
# 2                  4.7               3.2                1.3               0.2
# ...
# 149                5.9               3.0                5.1               1.8
# [150 rows x 4 columns]

df['Target'] = y
# print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
# 0                  5.1               3.5                1.4               0.2       0
# 1                  4.9               3.0                1.4               0.2       0
# ..                 ...               ...                ...               ...     ...
# 149                5.9               3.0                5.1               1.8       2
# [150 rows x 5 columns]

# save
path = './_save/'
joblib.dump(df, path + 'joblib/m19_correlation1.joblib')

# pickle.dump(df, open(path + 'pickle/m19_correlation1.pickle', 'wb'))
with open(path + 'pickle/m19_correlation1.pickle', 'wb') as f:
    pickle.dump(df, f)

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측