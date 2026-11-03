from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import datetime

#1. 데이터
path="./_data/kaggle_bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x)
x = scaler.transform(x)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델 구성
model = XGBRegressor()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)

print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))
# r2 : [0.35277212 0.3164866  0.33830118 0.31298316 0.34558314] 
# 평균 r2 : 0.33