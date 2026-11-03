from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

#1. 데이터
path = 'c:/study/_data/kaggle_santander/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x)
x = scaler.transform(x)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
# model1 = DecisionTreeClassifier(random_state=random_state)
# model2 = RandomForestClassifier(random_state=random_state)
# model3 = GradientBoostingClassifier(random_state=random_state)
model = XGBClassifier()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)

print("acc :", scores, "\n평균 acc :", round(np.mean(scores), 2))
# acc : [0.912925 0.913125 0.9117   0.9144   0.9117  ] 
# 평균 acc : 0.91